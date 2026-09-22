"""Synthetic integration checks. No research dataset or GPU is required."""
import json
from dataclasses import replace

import nrrd
import numpy as np
import pytest
import torch

import crop_individual_knots as crop
import resample_pine_knots_160x80x80_thesis_fit as resample
import prepare_sound_dead_dataset_11x80x80_thesis_fit as prepare
from sound_dead_config import TRAIN_TREE_NUMBERS, VALIDATION_TREE_NUMBERS, TEST_TREE_NUMBERS
from sound_dead_data import (
    PatchRecord, PatchDataset, balance_binary_training_records,
    normalize_ct_patch, split_records_by_fixed_trees, read_patch_index,
)
from sound_dead_metrics import boundary_error_metrics, fit_monotonic_transition
from sound_dead_model import SoundDeadCNN


def test_crop_resample_and_index(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    pith_root = tmp_path / 'pith'
    tree = source / 'PINE Tree 1_Finished'
    pith_dir = pith_root / 'Tree1'
    tree.mkdir(parents=True)
    pith_dir.mkdir(parents=True)
    shape = (72, 40, 20)
    mask = np.zeros(shape, np.uint8)
    mask[8:34, 18:24, 7:13] = 1
    mask[34:60, 18:24, 7:13] = 51
    pith = np.zeros(shape, np.uint8)
    pith[4:8, 18:22, :] = 1
    image = np.full(shape, 950, np.uint16)
    image[mask == 1] = 1500
    image[mask == 51] = 1800
    header = {'space': 'left-posterior-superior',
              'space directions': np.eye(3) * 0.5,
              'space origin': np.array([10., 20., 30.]), 'encoding': 'gzip'}
    for name, data in [('Drydisk01.1', image), ('Green Disk_01.1', image),
                       ('SEG_Drydisk01.1', mask)]:
        nrrd.write(str(tree / (name + '.nhdr')), data, header, index_order='F')
    nrrd.write(str(pith_dir / 'pith_SEG_Green Disk_01.1.nhdr'), pith, header, index_order='F')
    crop_root = tmp_path / 'crops'
    # monkeypatch restores mutable stage settings after the CLI calls.
    for module, names in [(crop, ['SOURCE_TREE_ROOT','PITH_ROOT','OUTPUT_ROOT','FIRST_TREE_NUMBER','LAST_TREE_NUMBER','NUMBER_OF_DISKS']),
                          (resample, ['SOURCE_ROOT','OUTPUT_ROOT','FIRST_TREE_NUMBER','LAST_TREE_NUMBER'])]:
        for name in names: monkeypatch.setattr(module, name, getattr(module, name))
    crop.cli(['--source-root', str(source), '--pith-root', str(pith_root),
              '--output-root', str(crop_root), '--first-tree', '1', '--last-tree', '1', '--disks', '1'])
    mask_file = next(crop_root.rglob('Mask_*.nhdr'))
    cropped, cropped_header = nrrd.read(str(mask_file), index_order='F')
    assert set(np.unique(cropped)) == {0, 1, 51, 150}
    assert np.all(np.any(cropped == 150, axis=(0, 1)))
    # Five extra slices on each side, not five total.
    assert cropped.shape[2] == 16
    assert cropped_header['space origin'][2] == 31.
    with pytest.raises(FileExistsError):
        crop.cli(['--source-root', str(source), '--pith-root', str(pith_root), '--output-root', str(crop_root)])

    blocks = tmp_path / 'blocks'
    resample.cli(['--source-root', str(crop_root), '--output-root', str(blocks), '--first-tree','1','--last-tree','1'])
    block_path = next(blocks.rglob('Mask_*.nhdr'))
    block, block_header = nrrd.read(str(block_path), index_order='F')
    assert block.shape == (160, 80, 80)
    assert {1, 51, 150}.issubset(set(np.unique(block)))
    transform = json.loads(next(blocks.rglob('Transform_*.json')).read_text())
    # Saved coordinate map must agree with the NHDR world coordinate geometry.
    point = np.array([12., 40., 40.])
    matrix = np.asarray(transform['output_to_original_source_index_matrix'])
    offset = np.asarray(transform['output_to_original_source_index_offset'])
    source_index = matrix @ point + offset
    world_source = cropped_header['space origin'] + source_index @ cropped_header['space directions']
    world_output = block_header['space origin'] + point @ block_header['space directions']
    np.testing.assert_allclose(world_source, world_output, atol=1e-6)

    monkeypatch.setattr(prepare, 'BLOCK_ROOT', str(blocks))
    sample = prepare.paths_for_knot(1, 1, 1, str(block_path.parent))
    rows, summary = prepare.rows_for_knot(sample)
    assert rows and {r['label'] for r in rows} == {0, 1}
    assert all(r['radial_stop_index_exclusive'] - r['radial_start_index'] == 11 for r in rows)
    assert all(r['label'] == int(r['radial_center_index'] >= r['transition_index_first_dead']) for r in rows)
    index = tmp_path / 'index.csv'
    prepare.write_csv(str(index), rows, prepare.INDEX_FIELDS)
    records = read_patch_index(str(index))
    ds = PatchDataset(records, block_root=str(blocks), image_state='wet',
                          augment_training=False, volume_cache_size=1)
    item = ds[0]
    assert item['image'].shape == (1, 11, 80, 80)
    assert torch.isfinite(item['image']).all()


def record(tree):
    split = 'train' if tree in TRAIN_TREE_NUMBERS else 'validation' if tree in VALIDATION_TREE_NUMBERS else 'test'
    label = tree % 2
    return PatchRecord(str(tree), split, tree, 1, 1, 10, 5, 16, 8.,
                       10 if label else 12, 8. if label else 10., 1., label,
                       'wet.nhdr', 'dry.nhdr', 'mask.nhdr', 'transform.json')


def test_fixed_tree_split_rejects_leakage():
    records = [record(tree) for tree in range(1,25)]
    train, val, test = split_records_by_fixed_trees(records)
    assert len(train) == 18 and len(val) == len(test) == 3
    wrong = [replace(r, split='train') if r.tree_number == 2 else r for r in records]
    with pytest.raises(ValueError, match='fixed tree split'): split_records_by_fixed_trees(wrong)
    with pytest.raises(ValueError, match='Missing'): split_records_by_fixed_trees(records[:-1])
    balanced = balance_binary_training_records(train, seed=42)
    assert sum(r.label == 0 for r in balanced) == sum(r.label == 1 for r in balanced)
    assert {r.tree_number for r in balanced}.issubset(set(TRAIN_TREE_NUMBERS))


def test_network_forward_and_backward():
    torch.set_num_threads(2)
    model = SoundDeadCNN()
    images = torch.randn(2,1,11,80,80)
    logits = model(images)
    assert logits.shape == (2,)
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0.,1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_normalisation_and_boundary_conventions():
    patch = np.zeros((11,80,80), dtype=np.float32)
    patch[0,0,:3] = [1,2,3]
    norm = normalize_ct_patch(patch)
    assert np.all(norm[patch == 0] == 0)
    assert abs(norm[patch != 0].mean()) < 1e-6
    assert fit_monotonic_transition([5,11,17],[0,0,0],sound_only_boundary=18)[0] == 18
    assert fit_monotonic_transition([5,11,17],[1,1,1],sound_only_boundary=18)[0] == 5
    # Preserve the supplied count rule even for inconsistent predictions.
    assert fit_monotonic_transition([5,11,17,23,29],[0,1,0,1,1],sound_only_boundary=30)[0] == 17
    errors = boundary_error_metrics([-3, 4])
    assert errors['mae_mm'] == 3.5
    assert errors['rmse_mm'] == pytest.approx(np.sqrt(12.5))


def test_train_and_evaluate_commands(tmp_path):
    """Exercise real entry points on a tiny synthetic fixture, not research data."""
    import csv
    import os
    from pathlib import Path
    import subprocess
    import sys
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    blocks = tmp_path / 'blocks'
    blocks.mkdir()
    rng = np.random.default_rng(42)
    phantom = rng.integers(900, 1800, size=(160,80,80), dtype=np.uint16)
    nrrd.write(str(blocks / 'phantom.nhdr'), phantom, {'encoding':'gzip'}, index_order='F')
    # One shared phantom keeps this software integration fixture small.
    # It is not an independent training/test dataset or a performance experiment.
    rows = []
    for tree in range(1,25):
        for center in (6,12):
            row = dict.fromkeys(prepare.INDEX_FIELDS, '')
            row.update(sample_id=f'T{tree:02d}_R{center}', split=record(tree).split,
                       tree=tree, disk=1, knot_id=1, radial_center_index=center,
                       radial_start_index=center-5, radial_stop_index_exclusive=center+6,
                       radial_mm_from_pith=center-2, transition_index_first_dead=9,
                       transition_mm_from_pith=7, effective_radial_spacing_mm=1,
                       label=int(center>=9), label_name='dead' if center>=9 else 'sound',
                       wet_relpath='phantom.nhdr', dry_relpath='phantom.nhdr',
                       mask_relpath='unused.nhdr', transform_relpath='unused.json')
            rows.append(row)
    index = tmp_path / 'index.csv'
    prepare.write_csv(str(index), rows, prepare.INDEX_FIELDS)
    results = tmp_path / 'results'
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    command = [sys.executable, str(scripts/'train_sound_dead.py'),
               '--block-root',str(blocks),'--index-csv',str(index),'--output-dir',str(results),
               '--epochs','1','--num-workers','0','--no-amp','--batch-size','6']
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (results/'best_model.pt').is_file()
    evaluation = results/'evaluation'
    command = [sys.executable, str(scripts/'evaluate_sound_dead.py'),
               '--block-root',str(blocks),'--index-csv',str(index),
               '--checkpoint',str(results/'best_model.pt'),'--output-dir',str(evaluation),
               '--num-workers','0','--no-amp','--timing-repeats','1','--timing-warmup','0']
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    summary = json.loads((evaluation/'evaluation_summary.json').read_text())
    assert summary['test_trees'] == [2,11,23]
    assert summary['patch_metrics']['samples'] == 6
    assert summary['transition_metrics']['timing']['knots'] == 3
    assert (evaluation/'transition_status_confusion_matrix.png').is_file()
