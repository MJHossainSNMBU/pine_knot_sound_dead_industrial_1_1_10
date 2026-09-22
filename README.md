# Pine knot sound and dead classification

A reproducible research workflow for pith guided knot cropping, physical resampling, sound and dead patch classification, boundary estimation and visual inspection of Pine CT data.

This is an adaptation of **Improving knot segmentation using Deep Learning techniques**, by Giovannini and colleagues. It uses the study input geometry with a conservative fitting policy and a classifier reconstructed using the accompanying thesis. It is not the authors' released implementation or an exact replication. Full log segmentation and knot diameter estimation are outside this repository.

## What is included

| Stage | Script in `scripts` |
| --- | --- |
| 1. Crop knots with pith | `crop_individual_knots.py` |
| 2. Resample, rotate and fit | `resample_pine_knots_160x80x80_thesis_fit.py` |
| 3. Index labelled patches | `prepare_sound_dead_dataset_11x80x80_thesis_fit.py` |
| Inspect three resolutions | `view_pine_knots_large_physical.py` |
| Train the classifier | `train_sound_dead.py` |
| Evaluate classification, boundary and timing | `evaluate_sound_dead.py` |
| Check model structure | `smoke_test_sound_dead.py` |

All scripts accept `--help` except the structural smoke test. The three preprocessing stages are separate. The viewer is for inspection only.

## Installation

Use Python 3.10 or later in an isolated environment. Run commands from the repository root.

```bash
python -m venv .venv
```

On Linux activate with `source .venv/bin/activate`. In Windows PowerShell use `.\.venv\Scripts\Activate.ps1`.

Install the PyTorch build appropriate for your GPU using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
```

For the local desktop viewer also install:

```bash
python -m pip install -r requirements_viewer.txt
```

PyTorch and VTK are not needed for cropping or resampling. The index builder can run without PyTorch, although its optional loader check will then be skipped. The viewer needs a desktop display.

## Paths and input data

Defaults are under `work` in the current working directory. Set `WAIKNOT_PROJECT_ROOT` to use a different work directory, or pass paths directly with command line options. Source data and generated arrays should be stored outside the public repository.

Read [the input format](docs/data_format.md) before preprocessing. The scripts expect knot instance labels and registered wet and dry images, not a generic semantic live and dead mask.

## Run the three preprocessing stages

Provide the folder containing `PINE Tree 1_Finished` through `PINE Tree 24_Finished` as the source root. Provide the folder containing `Tree1` through `Tree24` pith masks as the pith root. Replace the example input paths below with your own.

```bash
python scripts/crop_individual_knots.py --source-root /path/to/pine --pith-root /path/to/pith --output-root work/Individual_Knot_Crops --first-tree 1 --last-tree 24
python scripts/resample_pine_knots_160x80x80_thesis_fit.py --source-root work/Individual_Knot_Crops --output-root work/Individual_Knot_Crops_160x80x80_thesis_fit
python scripts/prepare_sound_dead_dataset_11x80x80_thesis_fit.py --block-root work/Individual_Knot_Crops_160x80x80_thesis_fit --output-root work/Sound_Dead_Dataset_11x80x80_thesis_fit --fail-on-skipped
```

Every output root must be a new folder. The scripts protect existing outputs. For repeated experiments choose new output paths and pass them to downstream stages.

The cropper and resampler record individual failures in manifests and can continue processing other samples. Review their failure and missing tree counts before training. `--fail-on-skipped` makes the index builder reject skipped knots. Training independently requires index entries for all 24 trees.

If the 160 by 80 by 80 blocks and patch index already exist, skip preprocessing. Supply their locations with `--block-root` and `--index-csv`.

## Train and evaluate

The fixed split is 18 training trees, validation trees **5, 8, 20**, and test trees **2, 11, 23**. Test trees do not guide training, early stopping or checkpoint selection.

```bash
python scripts/train_sound_dead.py --output-dir work/results_wet
python scripts/evaluate_sound_dead.py --checkpoint work/results_wet/best_model.pt --output-dir work/results_wet/evaluation
```

These commands use wet images and the default block and index locations. For dry images pass `--image-state dry` to both commands and choose a distinct result directory. The evaluation threshold remains fixed at 0.5.

Transfer **both** the block folder and the patch index folder to the cluster. Patches are loaded from the blocks on demand, and are not stored inside the CSV file. Transform JSON files accompany the blocks.

Portable Slurm scripts are in `slurm`. See [cluster instructions](docs/cluster.md).

## View the knots

```bash
python scripts/view_pine_knots_large_physical.py --original-root work/Individual_Knot_Crops --block-root work/Individual_Knot_Crops_160x80x80_thesis_fit --windowed
```

Rows show the original crop, the optional older resampling, and the final study geometry block. Supply `--previous-root` to display your existing older 1 by 1 by 10 mm output. That older output is not required by this pipeline. Without it, the middle row displays a missing data message.

| Control | Action |
| --- | --- |
| Left mouse drag | Rotate the four panels in that row together |
| Middle mouse drag | Pan the row |
| Wheel or right mouse drag | Zoom |
| M | Toggle common physical scale and independent row fitting |
| 1, 2, 3 | Enlarge the selected row |
| 0 | Restore all rows |
| R | Refit visible rows |
| B | Toggle display crop and full stored volume |
| F | Toggle fullscreen |
| L, D, P | Toggle live, dead and pith visibility |
| Arrow keys or Space | Move between knots |
| Q or Escape | Close |

The display crop removes empty margins only in memory. Voxel spacing, directions and origin determine physical geometry. In common scale mode the same physical length occupies the same screen length. Independent row fitting can use different magnification, shown by scale bars.

## Validation

```bash
python -m pip install -r requirements_test.txt
python scripts/smoke_test_sound_dead.py
python -m pytest -q
```

Tests use synthetic data. They check the actual crop, resampling and patch indexing interfaces, coordinate mappings, tree split protection, the classifier and boundary metrics. They do not establish accuracy on the real dataset. GitHub Actions runs these checks on CPU. See [package validation](docs/validation.md) for the checks completed before packaging.



Replace the remote URL with your own. Add the agreed software license and project authors before public release. GitHub ignores neither sensitive data nor personal paths automatically, so the provided ignore rules exclude CT arrays, weights and logs.
