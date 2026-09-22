"""CSV indexed NHDR loading for the study style sound and dead experiment."""

from __future__ import annotations

import csv
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import nrrd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from sound_dead_config import (
    DEAD_CLASS,
    EXPECTED_TREE_NUMBERS,
    FULL_BLOCK_SHAPE,
    PATCH_SHAPE,
    SOUND_CLASS,
    TEST_TREE_NUMBERS,
    TRAIN_TREE_NUMBERS,
    VALIDATION_TREE_NUMBERS,
)


INDEX_ORDER = "F"


@dataclass(frozen=True)
class PatchRecord:
    sample_id: str
    split: str
    tree_number: int
    disk_number: int
    knot_id: int
    center_r: int
    start_r: int
    stop_r: int
    radial_mm_from_pith: float
    transition_r: int
    transition_mm_from_pith: float
    spacing_radial_mm: float
    label: int
    wet_relpath: str
    dry_relpath: str
    mask_relpath: str
    transform_relpath: str

    @property
    def knot_key(self) -> Tuple[int, int, int]:
        return self.tree_number, self.disk_number, self.knot_id

    def image_relpath(self, image_state: str) -> str:
        if image_state == "wet":
            return self.wet_relpath
        if image_state == "dry":
            return self.dry_relpath
        raise ValueError("image_state must be wet or dry.")


REQUIRED_INDEX_FIELDS = {
    "sample_id",
    "split",
    "tree",
    "disk",
    "knot_id",
    "radial_center_index",
    "radial_start_index",
    "radial_stop_index_exclusive",
    "radial_mm_from_pith",
    "transition_index_first_dead",
    "transition_mm_from_pith",
    "effective_radial_spacing_mm",
    "label",
    "wet_relpath",
    "dry_relpath",
    "mask_relpath",
    "transform_relpath",
}


def portable_path_join(root: str, relative_path: str) -> str:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe relative path in index: {relative_path}")
    return os.path.join(root, *pure.parts)


def _parse_record(row: Mapping[str, str], row_number: int) -> PatchRecord:
    try:
        record = PatchRecord(
            sample_id=row["sample_id"],
            split=row["split"].strip().lower(),
            tree_number=int(row["tree"]),
            disk_number=int(row["disk"]),
            knot_id=int(row["knot_id"]),
            center_r=int(row["radial_center_index"]),
            start_r=int(row["radial_start_index"]),
            stop_r=int(row["radial_stop_index_exclusive"]),
            radial_mm_from_pith=float(row["radial_mm_from_pith"]),
            transition_r=int(row["transition_index_first_dead"]),
            transition_mm_from_pith=float(row["transition_mm_from_pith"]),
            spacing_radial_mm=float(row["effective_radial_spacing_mm"]),
            label=int(row["label"]),
            wet_relpath=row["wet_relpath"],
            dry_relpath=row["dry_relpath"],
            mask_relpath=row["mask_relpath"],
            transform_relpath=row["transform_relpath"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid index row {row_number}: {error}") from error

    if record.split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split in row {row_number}: {record.split}")
    if record.label not in {SOUND_CLASS, DEAD_CLASS}:
        raise ValueError(f"Invalid label in row {row_number}: {record.label}")
    if record.stop_r - record.start_r != PATCH_SHAPE[0]:
        raise ValueError(f"Wrong radial patch width in row {row_number}.")
    if not 0 <= record.start_r < record.stop_r <= FULL_BLOCK_SHAPE[0]:
        raise ValueError(f"Invalid radial bounds in row {row_number}.")
    if record.center_r != (record.start_r + record.stop_r - 1) // 2:
        raise ValueError(f"Patch is not centred in row {row_number}.")
    if record.spacing_radial_mm <= 0.0:
        raise ValueError(f"Nonpositive radial spacing in row {row_number}.")
    expected_label = int(record.center_r >= record.transition_r)
    if record.label != expected_label:
        raise ValueError(
            f"Label and transition disagree in row {row_number}: "
            f"{record.sample_id}."
        )
    return record


def read_patch_index(index_csv: str) -> List[PatchRecord]:
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(f"Patch index does not exist: {index_csv}")

    records: List[PatchRecord] = []
    seen_sample_ids = set()
    with open(index_csv, "r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        missing = sorted(REQUIRED_INDEX_FIELDS - set(reader.fieldnames or ()))
        if missing:
            raise KeyError(f"Patch index is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            record = _parse_record(row, row_number)
            if record.sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id: {record.sample_id}")
            seen_sample_ids.add(record.sample_id)
            records.append(record)

    if not records:
        raise ValueError("Patch index contains no records.")
    records.sort(
        key=lambda item: (
            item.tree_number,
            item.disk_number,
            item.knot_id,
            item.center_r,
        )
    )
    validate_knot_metadata(records)
    return records


def validate_knot_metadata(records: Sequence[PatchRecord]) -> None:
    by_knot = group_records_by_knot(records)
    for key, knot_records in by_knot.items():
        transitions = {record.transition_r for record in knot_records}
        spacings = {round(record.spacing_radial_mm, 12) for record in knot_records}
        paths = {
            (record.wet_relpath, record.dry_relpath)
            for record in knot_records
        }
        centres = [record.center_r for record in knot_records]
        if len(transitions) != 1 or len(spacings) != 1 or len(paths) != 1:
            raise ValueError(f"Inconsistent metadata for knot {key}.")
        if len(centres) != len(set(centres)):
            raise ValueError(f"Duplicate radial centre for knot {key}.")


def split_records_by_fixed_trees(
    records: Sequence[PatchRecord],
) -> Tuple[List[PatchRecord], List[PatchRecord], List[PatchRecord]]:
    available_trees = {record.tree_number for record in records}
    expected_trees = set(EXPECTED_TREE_NUMBERS)
    missing = sorted(expected_trees - available_trees)
    unexpected = sorted(available_trees - expected_trees)
    if missing or unexpected:
        raise ValueError(
            "The index must contain exactly trees 1 through 24. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )

    expected_split = {}
    expected_split.update({tree: "train" for tree in TRAIN_TREE_NUMBERS})
    expected_split.update(
        {tree: "validation" for tree in VALIDATION_TREE_NUMBERS}
    )
    expected_split.update({tree: "test" for tree in TEST_TREE_NUMBERS})
    errors = [
        record
        for record in records
        if record.split != expected_split[record.tree_number]
    ]
    if errors:
        first = errors[0]
        raise ValueError(
            "CSV split disagrees with the fixed tree split. First record: "
            f"{first.sample_id}."
        )

    train = [record for record in records if record.tree_number in TRAIN_TREE_NUMBERS]
    validation = [
        record
        for record in records
        if record.tree_number in VALIDATION_TREE_NUMBERS
    ]
    test = [record for record in records if record.tree_number in TEST_TREE_NUMBERS]
    if not train or not validation or not test:
        raise RuntimeError("Train, validation, and test must all be nonempty.")
    return train, validation, test


def balance_binary_training_records(
    records: Sequence[PatchRecord], seed: int
) -> List[PatchRecord]:
    sound = [record for record in records if record.label == SOUND_CLASS]
    dead = [record for record in records if record.label == DEAD_CLASS]
    if not sound or not dead:
        raise ValueError("Training data must contain sound and dead samples.")
    target = min(len(sound), len(dead))
    rng = random.Random(seed)
    selected = rng.sample(sound, target) + rng.sample(dead, target)
    selected.sort(
        key=lambda item: (
            item.tree_number,
            item.disk_number,
            item.knot_id,
            item.center_r,
        )
    )
    return selected


def class_counts(records: Iterable[PatchRecord]) -> Dict[str, int]:
    values = list(records)
    sound = sum(record.label == SOUND_CLASS for record in values)
    dead = sum(record.label == DEAD_CLASS for record in values)
    return {"sound": int(sound), "dead": int(dead), "total": len(values)}


def unique_knot_count(records: Sequence[PatchRecord]) -> int:
    return len({record.knot_key for record in records})


def group_records_by_knot(
    records: Sequence[PatchRecord],
) -> Dict[Tuple[int, int, int], List[PatchRecord]]:
    grouped: Dict[Tuple[int, int, int], List[PatchRecord]] = defaultdict(list)
    for record in records:
        grouped[record.knot_key].append(record)
    for values in grouped.values():
        values.sort(key=lambda record: record.center_r)
    return dict(sorted(grouped.items()))


def normalize_ct_patch(image: np.ndarray) -> np.ndarray:
    """Z score nonzero CT voxels and preserve zero padding."""

    image = np.asarray(image, dtype=np.float32)
    if tuple(image.shape) != PATCH_SHAPE:
        raise ValueError(f"Patch shape is {image.shape}, expected {PATCH_SHAPE}.")
    if not np.all(np.isfinite(image)):
        raise ValueError("A patch contains nonfinite CT values.")

    nonzero = image != 0
    values = image[nonzero] if np.count_nonzero(nonzero) >= 2 else image.ravel()
    mean = float(np.mean(values))
    standard_deviation = max(float(np.std(values)), 1e-6)
    if np.count_nonzero(nonzero) >= 2:
        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[nonzero] = (image[nonzero] - mean) / standard_deviation
        return normalized
    return (image - mean) / standard_deviation


def load_block(block_root: str, relative_path: str) -> np.ndarray:
    full_path = portable_path_join(block_root, relative_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"NHDR block does not exist: {full_path}")
    block, _ = nrrd.read(full_path, index_order=INDEX_ORDER)
    block = np.asarray(block)
    if tuple(block.shape) != FULL_BLOCK_SHAPE:
        raise ValueError(
            f"Block shape is {block.shape}, expected {FULL_BLOCK_SHAPE}: "
            f"{full_path}"
        )
    if not np.all(np.isfinite(block)):
        raise ValueError(f"Block contains nonfinite values: {full_path}")
    return block


def extract_patch(block: np.ndarray, start_r: int, stop_r: int) -> np.ndarray:
    if tuple(block.shape) != FULL_BLOCK_SHAPE:
        raise ValueError(f"Block shape is {block.shape}, expected {FULL_BLOCK_SHAPE}.")
    patch = np.asarray(block[int(start_r):int(stop_r), :, :])
    if tuple(patch.shape) != PATCH_SHAPE:
        raise ValueError(f"Extracted shape is {patch.shape}, expected {PATCH_SHAPE}.")
    return patch


class PatchDataset(Dataset):
    """Lazily extract indexed 11 by 80 by 80 patches from NHDR blocks."""

    def __init__(
        self,
        records: Sequence[PatchRecord],
        *,
        block_root: str,
        image_state: str,
        augment_training: bool,
        volume_cache_size: int,
    ) -> None:
        if not records:
            raise ValueError("Dataset records cannot be empty.")
        if image_state not in {"wet", "dry"}:
            raise ValueError("image_state must be wet or dry.")
        if volume_cache_size < 1:
            raise ValueError("volume_cache_size must be positive.")
        self.records = list(records)
        self.block_root = block_root
        self.image_state = image_state
        self.augment_training = bool(augment_training)
        self.augmentation_factor = 3 if augment_training else 1
        self.volume_cache_size = int(volume_cache_size)
        self._volume_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records) * self.augmentation_factor

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_volume_cache"] = OrderedDict()
        return state

    def _load(self, relative_path: str) -> np.ndarray:
        cached = self._volume_cache.get(relative_path)
        if cached is not None:
            self._volume_cache.move_to_end(relative_path)
            return cached
        block = load_block(self.block_root, relative_path)
        self._volume_cache[relative_path] = block
        self._volume_cache.move_to_end(relative_path)
        while len(self._volume_cache) > self.volume_cache_size:
            self._volume_cache.popitem(last=False)
        return block

    def __getitem__(self, dataset_index: int) -> dict:
        record_index = dataset_index // self.augmentation_factor
        augmentation_index = dataset_index % self.augmentation_factor
        record = self.records[record_index]
        block = self._load(record.image_relpath(self.image_state))
        patch = np.array(
            extract_patch(block, record.start_r, record.stop_r),
            dtype=np.float32,
            order="C",
            copy=True,
        )

        augmentation = "original"
        if augmentation_index == 1:
            patch = np.flip(patch, axis=1)
            augmentation = "tangential_flip"
        elif augmentation_index == 2:
            patch = np.flip(patch, axis=2)
            augmentation = "longitudinal_flip"

        patch = normalize_ct_patch(patch)
        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(patch[None, ...], dtype=np.float32)
            ),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "sample_id": record.sample_id,
            "tree": record.tree_number,
            "disk": record.disk_number,
            "knot_id": record.knot_id,
            "center_r": record.center_r,
            "radial_mm_from_pith": record.radial_mm_from_pith,
            "augmentation": augmentation,
        }


class KnotGroupedBatchSampler(Sampler[List[int]]):
    """Shuffle knots while keeping each batch local to one NHDR volume."""

    def __init__(
        self,
        dataset: PatchDataset,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        record_indices: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            record_indices[record.knot_key].append(index)
        self.record_indices = dict(record_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        knot_keys = list(self.record_indices)
        rng.shuffle(knot_keys)
        factor = self.dataset.augmentation_factor
        for key in knot_keys:
            record_indices = list(self.record_indices[key])
            rng.shuffle(record_indices)
            dataset_indices = [
                record_index * factor + augmentation_index
                for record_index in record_indices
                for augmentation_index in range(factor)
            ]
            rng.shuffle(dataset_indices)
            for start in range(0, len(dataset_indices), self.batch_size):
                yield dataset_indices[start:start + self.batch_size]

    def __len__(self) -> int:
        factor = self.dataset.augmentation_factor
        return sum(
            (len(indices) * factor + self.batch_size - 1) // self.batch_size
            for indices in self.record_indices.values()
        )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    records: Sequence[PatchRecord],
    *,
    block_root: str,
    image_state: str,
    batch_size: int,
    num_workers: int,
    volume_cache_size: int,
    training: bool,
    seed: int,
) -> DataLoader:
    dataset = PatchDataset(
        records,
        block_root=block_root,
        image_state=image_state,
        augment_training=training,
        volume_cache_size=volume_cache_size,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "dataset": dataset,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    if training:
        sampler = KnotGroupedBatchSampler(dataset, batch_size, seed)
        return DataLoader(batch_sampler=sampler, **common)
    return DataLoader(
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )


def write_split_manifest(
    output_path: str,
    split_records_map: Mapping[str, Sequence[PatchRecord]],
) -> None:
    fields = (
        "selection",
        "tree",
        "disk",
        "knot_id",
        "center_r",
        "label",
        "label_name",
        "sample_id",
        "wet_relpath",
        "dry_relpath",
    )
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for selection, values in split_records_map.items():
            for record in values:
                writer.writerow(
                    {
                        "selection": selection,
                        "tree": record.tree_number,
                        "disk": record.disk_number,
                        "knot_id": record.knot_id,
                        "center_r": record.center_r,
                        "label": record.label,
                        "label_name": "sound" if record.label == 0 else "dead",
                        "sample_id": record.sample_id,
                        "wet_relpath": record.wet_relpath,
                        "dry_relpath": record.dry_relpath,
                    }
                )
