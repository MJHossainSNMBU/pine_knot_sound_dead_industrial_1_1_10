"""Prepare 11 by 80 by 80 sound and dead inputs from thesis-fit blocks.

This script uses the 160 by 80 by 80 blocks created by
resample_pine_knots_160x80x80_thesis_fit.py. It does not resample,
stretch, compress, or rewrite their image voxels. Instead, it creates a
portable CSV index with one row for every valid 11 by 80 by 80 radial
subvolume. ThesisFitSoundDeadDataset reads the indexed rows and extracts
PyTorch tensors only when training requests them.

Published geometry reproduced here:

    full knot block: 160 by 80 by 80 in radial, tangential, longitudinal order
    classifier input: 11 consecutive slices along the radial axis
    classifier target: status at the central radial slice

The study used subvolumes of 11 consecutive slices along the radial direction.
It used one position every six slices only for the coarse inference search,
followed by fine refinement. Six is therefore not used as the training stride
here.

The study derived labels from manual board measurements and extended them
monotonically along the radial direction. Our data do not contain those board
measurements. This implementation estimates one monotonic transition from the
live and dead labels in each resampled mask. The labels are therefore a
documented surrogate for applying the study method to this dataset.

The fixed tree split is retained from the earlier Pine experiments:

    validation trees: 5, 8, 20
    test trees: 2, 11, 23
    training trees: all remaining trees from 1 through 24

Run this file on the preparation computer to create the index. For Orion, transfer the
existing Individual_Knot_Crops_160x80x80_thesis_fit folder, the small
Sound_Dead_Dataset_11x80x80_thesis_fit folder created here, and this
script. Instantiate ThesisFitSoundDeadDataset with the Orion block root.
No NPZ or separate PT file is needed for every overlapping patch.

Required to prepare the index:

    pip install numpy pynrrd

Required to use the dataset for training:

    pip install torch
"""

from __future__ import annotations

import paths as paths

import csv
import json
import os
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

import nrrd
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Import placeholder so index preparation does not require PyTorch."""


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

FIRST_TREE_NUMBER = 1
LAST_TREE_NUMBER = 24

BLOCK_ROOT = str(paths.BLOCK_ROOT)
OUTPUT_ROOT = str(paths.DATASET_ROOT)

VALIDATION_TREES = (5, 8, 20)
TEST_TREES = (2, 11, 23)

BLOCK_SHAPE = (160, 80, 80)
PATCH_SHAPE = (11, 80, 80)
PATCH_HALF_WIDTH = PATCH_SHAPE[0] // 2

# A centre is indexed only when its radial slice contains knot tissue.
MINIMUM_KNOT_VOXELS_AT_CENTRE = 1

BACKGROUND_LABEL = 0
DEAD_LABEL_OFFSET = 50
PITH_LABEL = 150
CLEAR_WOOD_LABEL = 255

INDEX_ORDER = "F"
REFUSE_EXISTING_OUTPUT_ROOT = True
FAIL_IF_ANY_KNOT_IS_SKIPPED = False

INDEX_FILENAME = "patch_index_11x80x80_thesis_fit.csv"
SKIPPED_FILENAME = "skipped_knots_11x80x80_thesis_fit.csv"
METADATA_FILENAME = "dataset_metadata_11x80x80_thesis_fit.json"


INDEX_FIELDS = [
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
    "label_name",
    "center_live_voxels",
    "center_dead_voxels",
    "center_dead_fraction",
    "transition_fit_error_voxels",
    "wet_relpath",
    "dry_relpath",
    "mask_relpath",
    "transform_relpath",
]

SKIPPED_FIELDS = [
    "tree",
    "disk",
    "knot_id",
    "reason",
]


@dataclass(frozen=True)
class KnotFiles:
    tree_number: int
    disk_number: int
    knot_id: int
    wet_image: str
    dry_image: str
    shared_mask: str
    transform_json: str


def validate_settings() -> None:
    if FIRST_TREE_NUMBER > LAST_TREE_NUMBER:
        raise ValueError(
            "FIRST_TREE_NUMBER cannot be greater than LAST_TREE_NUMBER."
        )

    validation = set(int(value) for value in VALIDATION_TREES)
    test = set(int(value) for value in TEST_TREES)
    if validation & test:
        raise ValueError("Validation and test tree sets overlap.")

    valid_range = set(range(FIRST_TREE_NUMBER, LAST_TREE_NUMBER + 1))
    outside = sorted((validation | test) - valid_range)
    if outside:
        raise ValueError(f"Split trees outside the configured range: {outside}")

    if BLOCK_SHAPE != (160, 80, 80):
        raise ValueError(
            "BLOCK_SHAPE must remain 160 by 80 by 80 for this "
            "replication."
        )
    if PATCH_SHAPE != (11, 80, 80):
        raise ValueError(
            "PATCH_SHAPE must remain 11 by 80 by 80 for this "
            "replication."
        )
    if PATCH_SHAPE[0] % 2 == 0:
        raise ValueError("The radial patch width must be odd.")
    if MINIMUM_KNOT_VOXELS_AT_CENTRE <= 0:
        raise ValueError(
            "MINIMUM_KNOT_VOXELS_AT_CENTRE must be positive."
        )

def numbered_thesis_fit_subdirectories(
    parent: str,
    prefix: str,
) -> List[Tuple[int, str]]:
    pattern = re.compile(
        rf"^{re.escape(prefix)}(\d+)_thesis_fit$",
        re.IGNORECASE,
    )
    matches: List[Tuple[int, str]] = []
    if not os.path.isdir(parent):
        return matches

    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        match = pattern.fullmatch(name)
        if match and os.path.isdir(path):
            matches.append((int(match.group(1)), path))

    return sorted(matches, key=lambda item: item[0])


def paths_for_knot(
    tree_number: int,
    disk_number: int,
    knot_id: int,
    knot_directory: str,
) -> KnotFiles:
    identity = (
        f"thesis_fit_Tree{tree_number:02d}_Disk{disk_number:02d}_"
        f"Knot{knot_id:02d}"
    )
    return KnotFiles(
        tree_number=tree_number,
        disk_number=disk_number,
        knot_id=knot_id,
        wet_image=os.path.join(knot_directory, f"Wet_{identity}.nhdr"),
        dry_image=os.path.join(knot_directory, f"Dry_{identity}.nhdr"),
        shared_mask=os.path.join(knot_directory, f"Mask_{identity}.nhdr"),
        transform_json=os.path.join(
            knot_directory,
            f"Transform_{identity}.json",
        ),
    )


def discover_knots() -> Tuple[List[KnotFiles], List[int]]:
    samples: List[KnotFiles] = []
    missing_trees: List[int] = []

    for tree_number in range(FIRST_TREE_NUMBER, LAST_TREE_NUMBER + 1):
        tree_directory = os.path.join(
            BLOCK_ROOT,
            f"Tree{tree_number:02d}_thesis_fit",
        )
        if not os.path.isdir(tree_directory):
            missing_trees.append(tree_number)
            continue

        for disk_number, disk_directory in (
            numbered_thesis_fit_subdirectories(
                tree_directory,
                "Disk",
            )
        ):
            for knot_id, knot_directory in (
                numbered_thesis_fit_subdirectories(
                    disk_directory,
                    "Knot",
                )
            ):
                samples.append(
                    paths_for_knot(
                        tree_number,
                        disk_number,
                        knot_id,
                        knot_directory,
                    )
                )

    return samples, missing_trees


def detached_raw_path(nhdr_path: str) -> str:
    if not nhdr_path.lower().endswith(".nhdr"):
        raise ValueError(f"Expected an NHDR path: {nhdr_path}")
    return nhdr_path[:-5] + ".raw.gz"


def required_file_paths(sample: KnotFiles) -> List[str]:
    paths = [
        sample.wet_image,
        detached_raw_path(sample.wet_image),
        sample.dry_image,
        detached_raw_path(sample.dry_image),
        sample.shared_mask,
        detached_raw_path(sample.shared_mask),
        sample.transform_json,
    ]
    return paths


def portable_relative_path(path: str, root: str) -> str:
    relative = os.path.relpath(path, root)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise ValueError(f"Path lies outside BLOCK_ROOT: {path}")
    return relative.replace("\\", "/")


def portable_path_join(root: str, relative: str) -> str:
    pure_path = PurePosixPath(relative)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"Unsafe relative path in patch index: {relative}")
    return os.path.join(root, *pure_path.parts)


def split_for_tree(tree_number: int) -> str:
    if tree_number in TEST_TREES:
        return "test"
    if tree_number in VALIDATION_TREES:
        return "validation"
    return "train"


def header_shape(path: str) -> Tuple[int, int, int]:
    header = nrrd.read_header(path)
    if "sizes" not in header:
        raise ValueError(f"NHDR has no sizes field: {path}")
    sizes = tuple(int(value) for value in np.asarray(header["sizes"]).tolist())
    if len(sizes) != 3:
        raise ValueError(f"NHDR is not three dimensional: {path}")
    return sizes


def load_transform_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as input_file:
        metadata = json.load(input_file)

    shape = tuple(int(value) for value in metadata.get(
        "target_shape_voxels",
        [],
    ))
    if shape != BLOCK_SHAPE:
        raise ValueError(
            f"Transform target shape is {shape}, expected {BLOCK_SHAPE}."
        )

    method = (
        metadata.get("replication_target", {})
        .get("method", "")
    )
    if method != "study_shape_with_thesis_fit_policy":
        raise ValueError(
            "Transform metadata is not from the 160 by 80 by 80 thesis-fit "
            "study resampler."
        )

    return metadata


def monotonic_transition_index(
    live_counts: np.ndarray,
    dead_counts: np.ndarray,
    active_minimum: int,
    active_maximum: int,
) -> Tuple[int, int]:
    """Fit sound before the boundary and dead from the boundary onward."""

    live_prefix = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(live_counts, dtype=np.int64))
    )
    dead_prefix = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(dead_counts, dtype=np.int64))
    )

    boundaries = np.arange(
        active_minimum,
        active_maximum + 2,
        dtype=int,
    )
    errors = (
        dead_prefix[boundaries] - dead_prefix[active_minimum]
        + live_prefix[active_maximum + 1]
        - live_prefix[boundaries]
    )
    minimum_error = int(np.min(errors))
    best = boundaries[errors == minimum_error]
    transition = int(np.rint(np.median(best)))
    return transition, minimum_error


def rows_for_knot(sample: KnotFiles) -> Tuple[List[dict], dict]:
    missing = [path for path in required_file_paths(sample) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(missing)
        )

    for image_path in (sample.wet_image, sample.dry_image):
        shape = header_shape(image_path)
        if shape != BLOCK_SHAPE:
            raise ValueError(
                f"Image shape is {shape}, expected {BLOCK_SHAPE}: "
                f"{image_path}"
            )

    mask, _ = nrrd.read(sample.shared_mask, index_order=INDEX_ORDER)
    if mask.ndim != 3 or tuple(mask.shape) != BLOCK_SHAPE:
        raise ValueError(
            f"Mask shape is {mask.shape}, expected {BLOCK_SHAPE}."
        )
    if not np.issubdtype(mask.dtype, np.integer):
        rounded = np.rint(mask)
        if not np.allclose(mask, rounded, rtol=0.0, atol=1e-6):
            raise ValueError("Mask contains noninteger values.")
        mask = rounded.astype(np.int32)

    metadata = load_transform_metadata(sample.transform_json)
    pith_index = int(metadata.get("radial_pith_output_index", -1))
    if not 0 <= pith_index < BLOCK_SHAPE[0]:
        raise ValueError("Transform contains an invalid radial pith index.")

    spacing = metadata.get("effective_output_spacing_mm", [])
    if len(spacing) != 3:
        raise ValueError("Transform has no valid effective output spacing.")
    radial_spacing_mm = float(spacing[0])
    if not np.isfinite(radial_spacing_mm) or radial_spacing_mm <= 0.0:
        raise ValueError("Effective radial spacing must be positive.")

    live_label = int(sample.knot_id)
    dead_label = int(sample.knot_id + DEAD_LABEL_OFFSET)
    live_counts = np.count_nonzero(mask == live_label, axis=(1, 2)).astype(
        np.int64
    )
    dead_counts = np.count_nonzero(mask == dead_label, axis=(1, 2)).astype(
        np.int64
    )
    knot_counts = live_counts + dead_counts
    active_indices = np.flatnonzero(knot_counts > 0)
    if active_indices.size == 0:
        raise ValueError("Mask contains no live or dead voxels for this knot.")

    active_minimum = int(active_indices[0])
    active_maximum = int(active_indices[-1])
    transition_index, fit_error = monotonic_transition_index(
        live_counts,
        dead_counts,
        active_minimum,
        active_maximum,
    )

    first_valid_center = max(
        PATCH_HALF_WIDTH,
        pith_index,
        active_minimum,
    )
    last_valid_center = min(
        BLOCK_SHAPE[0] - 1 - PATCH_HALF_WIDTH,
        active_maximum,
    )
    if first_valid_center > last_valid_center:
        raise ValueError("Knot has no valid centre for an 11 slice subblock.")

    split = split_for_tree(sample.tree_number)
    transition_mm = (
        float(transition_index - pith_index) * radial_spacing_mm
    )
    wet_relpath = portable_relative_path(sample.wet_image, BLOCK_ROOT)
    dry_relpath = portable_relative_path(sample.dry_image, BLOCK_ROOT)
    mask_relpath = portable_relative_path(sample.shared_mask, BLOCK_ROOT)
    transform_relpath = portable_relative_path(
        sample.transform_json,
        BLOCK_ROOT,
    )

    rows: List[dict] = []
    for center in range(first_valid_center, last_valid_center + 1):
        if int(knot_counts[center]) < MINIMUM_KNOT_VOXELS_AT_CENTRE:
            continue

        label = int(center >= transition_index)
        live_at_center = int(live_counts[center])
        dead_at_center = int(dead_counts[center])
        total_at_center = live_at_center + dead_at_center
        dead_fraction = (
            float(dead_at_center) / float(total_at_center)
            if total_at_center > 0
            else 0.0
        )
        start = center - PATCH_HALF_WIDTH
        stop = center + PATCH_HALF_WIDTH + 1

        rows.append(
            {
                "sample_id": (
                    f"T{sample.tree_number:02d}_D{sample.disk_number:02d}_"
                    f"K{sample.knot_id:02d}_R{center:03d}"
                ),
                "split": split,
                "tree": sample.tree_number,
                "disk": sample.disk_number,
                "knot_id": sample.knot_id,
                "radial_center_index": center,
                "radial_start_index": start,
                "radial_stop_index_exclusive": stop,
                "radial_mm_from_pith": (
                    float(center - pith_index) * radial_spacing_mm
                ),
                "transition_index_first_dead": transition_index,
                "transition_mm_from_pith": transition_mm,
                "effective_radial_spacing_mm": radial_spacing_mm,
                "label": label,
                "label_name": "dead" if label == 1 else "sound",
                "center_live_voxels": live_at_center,
                "center_dead_voxels": dead_at_center,
                "center_dead_fraction": dead_fraction,
                "transition_fit_error_voxels": fit_error,
                "wet_relpath": wet_relpath,
                "dry_relpath": dry_relpath,
                "mask_relpath": mask_relpath,
                "transform_relpath": transform_relpath,
            }
        )

    if not rows:
        raise ValueError("No indexed subblocks remain after centre filtering.")

    summary = {
        "tree": sample.tree_number,
        "disk": sample.disk_number,
        "knot_id": sample.knot_id,
        "split": split,
        "pith_index": pith_index,
        "active_radial_minimum": active_minimum,
        "active_radial_maximum": active_maximum,
        "transition_index_first_dead": transition_index,
        "transition_mm_from_pith": transition_mm,
        "transition_fit_error_voxels": fit_error,
        "live_voxels": int(np.sum(live_counts)),
        "dead_voxels": int(np.sum(dead_counts)),
        "patches": len(rows),
        "sound_patches": sum(int(row["label"]) == 0 for row in rows),
        "dead_patches": sum(int(row["label"]) == 1 for row in rows),
    }
    return rows, summary


def write_csv(path: str, rows: List[dict], fields: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def count_rows(rows: Sequence[dict], key: str) -> Dict[str, int]:
    counts = Counter(str(row[key]) for row in rows)
    return dict(sorted(counts.items()))


def build_index() -> Tuple[str, str, str]:
    validate_settings()
    if REFUSE_EXISTING_OUTPUT_ROOT and os.path.exists(OUTPUT_ROOT):
        raise FileExistsError(
            "Dataset output already exists and will not be overwritten: "
            f"{OUTPUT_ROOT}. Choose a new folder such as "
            "Sound_Dead_Dataset_11x80x80_thesis_fit_v2."
        )

    samples, missing_trees = discover_knots()
    if not samples:
        raise FileNotFoundError(
            "No thesis-fit study knot blocks were found below "
            f"{BLOCK_ROOT}."
        )

    rows: List[dict] = []
    skipped: List[dict] = []
    knot_summaries: List[dict] = []

    for index, sample in enumerate(samples, start=1):
        identity = (
            f"Tree {sample.tree_number:02d} | "
            f"Disk {sample.disk_number:02d} | "
            f"Knot {sample.knot_id:02d}"
        )
        try:
            knot_rows, summary = rows_for_knot(sample)
            rows.extend(knot_rows)
            knot_summaries.append(summary)
            print(
                f"{index} of {len(samples)} | {identity} | "
                f"{len(knot_rows)} patches"
            )
        except Exception as error:
            skipped.append(
                {
                    "tree": sample.tree_number,
                    "disk": sample.disk_number,
                    "knot_id": sample.knot_id,
                    "reason": str(error),
                }
            )
            print(
                f"{index} of {len(samples)} | {identity} | SKIPPED | {error}"
            )

    if skipped and FAIL_IF_ANY_KNOT_IS_SKIPPED:
        raise RuntimeError(
            f"{len(skipped)} knots could not be indexed. Nothing was written."
        )
    if not rows:
        raise RuntimeError(
            "No 11 by 80 by 80 patches were indexed. Nothing was written."
        )

    os.makedirs(OUTPUT_ROOT, exist_ok=False)
    index_path = os.path.join(OUTPUT_ROOT, INDEX_FILENAME)
    skipped_path = os.path.join(OUTPUT_ROOT, SKIPPED_FILENAME)
    metadata_path = os.path.join(OUTPUT_ROOT, METADATA_FILENAME)

    write_csv(index_path, rows, INDEX_FIELDS)
    write_csv(skipped_path, skipped, SKIPPED_FIELDS)

    metadata = {
        "schema_version": 2,
        "replication_target": (
            "study_sound_dead_stage_on_thesis_fit_blocks"
        ),
        "source_block_method": "study_shape_with_thesis_fit_policy",
        "storage": {
            "method": "lazy_nhdr_patch_index",
            "patch_voxels_are_not_duplicated": True,
            "index_file": INDEX_FILENAME,
            "source_block_root_when_prepared": BLOCK_ROOT,
            "portable_paths": True,
        },
        "published_geometry": {
            "full_block_shape_r_t_z": list(BLOCK_SHAPE),
            "subblock_shape_r_t_z": list(PATCH_SHAPE),
            "target_position": "central_radial_slice",
            "coarse_inference_step_radial_slices": 6,
            "training_index_step_radial_slices": 1,
            "study_reported_inferences_per_knot": 23,
        },
        "labels": {
            "sound": 0,
            "dead": 1,
            "method": "monotonic_transition_fit_from_resampled_mask",
            "note": (
                "The published study used manual board measurements. "
                "Mask-derived labels are a surrogate for this dataset."
            ),
        },
        "split": {
            "policy": "held_out_trees",
            "validation_trees": list(VALIDATION_TREES),
            "test_trees": list(TEST_TREES),
            "training_trees": sorted(
                set(range(FIRST_TREE_NUMBER, LAST_TREE_NUMBER + 1))
                - set(VALIDATION_TREES)
                - set(TEST_TREES)
            ),
            "note": (
                "This leakage-resistant tree split is specific to our Pine "
                "experiment. The study used measured intersections rather "
                "than completely held-out trees."
            ),
        },
        "counts": {
            "discovered_knots": len(samples),
            "indexed_knots": len(knot_summaries),
            "skipped_knots": len(skipped),
            "patches": len(rows),
            "patches_by_split": count_rows(rows, "split"),
            "patches_by_label": count_rows(rows, "label_name"),
            "missing_tree_folders": missing_trees,
        },
        "knot_summaries": knot_summaries,
    }
    with open(metadata_path, "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    return index_path, skipped_path, metadata_path


class ThesisFitSoundDeadDataset(Dataset):
    """Lazy PyTorch dataset for 11 by 80 by 80 thesis-fit patches."""

    def __init__(
        self,
        index_csv: str,
        data_root: str,
        split: Optional[str] = None,
        image_state: str = "wet",
        normalization: str = "none",
        cache_size: int = 8,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required to use "
                "ThesisFitSoundDeadDataset."
            )
        if split not in (None, "train", "validation", "test"):
            raise ValueError(
                "split must be train, validation, test, or None."
            )
        if image_state not in ("wet", "dry"):
            raise ValueError("image_state must be wet or dry.")
        if normalization not in ("none", "patch_zscore"):
            raise ValueError(
                "normalization must be none or patch_zscore."
            )
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative.")

        with open(index_csv, "r", newline="", encoding="utf-8-sig") as input_file:
            all_rows = [dict(row) for row in csv.DictReader(input_file)]
        self.rows = (
            all_rows
            if split is None
            else [row for row in all_rows if row.get("split") == split]
        )
        if not self.rows:
            raise ValueError("No rows match the requested dataset split.")

        self.data_root = data_root
        self.image_state = image_state
        self.normalization = normalization
        self.cache_size = int(cache_size)
        self._volume_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_volume_cache"] = OrderedDict()
        return state

    def _load_volume(self, relative_path: str) -> np.ndarray:
        full_path = portable_path_join(self.data_root, relative_path)
        if full_path in self._volume_cache:
            volume = self._volume_cache.pop(full_path)
            self._volume_cache[full_path] = volume
            return volume

        volume, _ = nrrd.read(full_path, index_order=INDEX_ORDER)
        if volume.ndim != 3 or tuple(volume.shape) != BLOCK_SHAPE:
            raise ValueError(
                f"Volume shape is {volume.shape}, expected "
                f"{BLOCK_SHAPE}: {full_path}"
            )
        volume = np.asarray(volume)

        if self.cache_size > 0:
            self._volume_cache[full_path] = volume
            while len(self._volume_cache) > self.cache_size:
                self._volume_cache.popitem(last=False)
        return volume

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path_field = f"{self.image_state}_relpath"
        volume = self._load_volume(row[path_field])
        start = int(row["radial_start_index"])
        stop = int(row["radial_stop_index_exclusive"])
        patch = np.array(
            volume[start:stop, :, :],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        if tuple(patch.shape) != PATCH_SHAPE:
            raise ValueError(
                f"Indexed patch shape is {patch.shape}, expected "
                f"{PATCH_SHAPE}."
            )

        if self.normalization == "patch_zscore":
            mean = float(np.mean(patch))
            standard_deviation = float(np.std(patch))
            patch -= mean
            patch /= max(standard_deviation, 1e-6)

        return {
            "image": torch.from_numpy(patch).unsqueeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": row["sample_id"],
            "tree": int(row["tree"]),
            "disk": int(row["disk"]),
            "knot_id": int(row["knot_id"]),
            "radial_center_index": int(row["radial_center_index"]),
            "radial_mm_from_pith": float(row["radial_mm_from_pith"]),
        }


def verify_dataset_loader(index_path: str) -> None:
    if torch is None:
        print("PyTorch is not installed. The CSV index was still created.")
        return

    dataset = ThesisFitSoundDeadDataset(
        index_csv=index_path,
        data_root=BLOCK_ROOT,
        split="train",
        image_state="wet",
        normalization="none",
        cache_size=1,
    )
    item = dataset[0]
    expected_tensor_shape = (1, *PATCH_SHAPE)
    if tuple(item["image"].shape) != expected_tensor_shape:
        raise ValueError(
            f"Loader produced {tuple(item['image'].shape)}, expected "
            f"{expected_tensor_shape}."
        )
    print(
        "PyTorch loader check passed: "
        f"{item['sample_id']} has tensor shape "
        f"{tuple(item['image'].shape)}."
    )


def main() -> None:
    index_path, skipped_path, metadata_path = build_index()
    verify_dataset_loader(index_path)

    with open(index_path, "r", newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    print("\nFinished")
    print(f"Indexed patches: {len(rows)}")
    print(f"By split: {count_rows(rows, 'split')}")
    print(f"By label: {count_rows(rows, 'label_name')}")
    print(f"Patch index: {index_path}")
    print(f"Skipped knot report: {skipped_path}")
    print(f"Dataset metadata: {metadata_path}")
    print("No image voxels were duplicated into patch files.")


def cli(argv=None):
    import argparse
    global BLOCK_ROOT, OUTPUT_ROOT, FAIL_IF_ANY_KNOT_IS_SKIPPED
    parser = argparse.ArgumentParser(description='Index 11 x 80 x 80 patches for the fixed 24 tree experiment.')
    parser.add_argument('--block-root', default=BLOCK_ROOT)
    parser.add_argument('--output-root', default=OUTPUT_ROOT)
    parser.add_argument('--fail-on-skipped', action='store_true')
    args = parser.parse_args(argv)
    BLOCK_ROOT, OUTPUT_ROOT = args.block_root, args.output_root
    FAIL_IF_ANY_KNOT_IS_SKIPPED = args.fail_on_skipped
    paths.require_input_directory(BLOCK_ROOT)
    paths.require_new_output(OUTPUT_ROOT)
    main()


if __name__ == "__main__":
    cli()
