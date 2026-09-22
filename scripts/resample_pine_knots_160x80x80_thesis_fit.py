"""Prepare 160 by 80 by 80 knot blocks with thesis-style fitting.

For every pine knot crop, this script performs the following operations:

1. Convert the original 0.5 mm isotropic crop to the 1 by 1 by 10 mm CT grid
   stated in Giovannini et al.'s published study.
2. Estimate the outward knot direction from the pith and the knot mask.
3. Rotate the knot in the cross sectional plane so the output axes represent
   radial, tangential, and longitudinal directions.
4. Produce one fixed 160 by 80 by 80 block for the dry image, wet image, and
   shared mask.
5. Keep the 1 by 1 by 10 mm sampling step when a knot fits and add padding.
   Only an axis that is too large is downsampled enough to fit the box. No
   small knot is stretched to fill the box.
6. Save all rotation, scaling, padding, and inverse mapping information in a
   JSON file beside every knot and in one combined CSV manifest.
7. If inverse mask sampling removes every live voxel, forward map genuine
   source live voxels into the output grid and restore the best supported
   non-pith voxel. Image values and geometric transforms are not changed.

This is deliberately a hybrid experiment. It uses the published study block
shape and later 11 slice classifier width, but it uses the conservative fitting
policy from the thesis replication. It is not the study's unpublished scaling
procedure. The distinction is stored in every transform JSON file.

The input files are not modified. Use --source-root and --output-root.
Output roots must be new folders containing .

Array axis order in every final output is:

    axis 0 = radial
    axis 1 = tangential
    axis 2 = longitudinal

Required packages:

    pip install numpy scipy pynrrd
"""

from __future__ import annotations

import paths as paths

import copy
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import nrrd
import numpy as np
from scipy.ndimage import map_coordinates


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

FIRST_TREE_NUMBER = 1
LAST_TREE_NUMBER = 24

SOURCE_ROOT = str(paths.CROP_ROOT)
OUTPUT_ROOT = str(paths.BLOCK_ROOT)

# The original cropper stores the through stem direction on NumPy axis 2.
# Local alignment below currently requires this value to remain 2.
STEM_AXIS = 2

EXPECTED_SOURCE_SPACING_MM = (0.5, 0.5, 0.5)
INDUSTRY_SPACING_MM = (1.0, 1.0, 10.0)
SPACING_TOLERANCE_MM = 1e-5

# Published fixed block in radial, tangential, longitudinal order.
TARGET_BLOCK_SHAPE = (160, 80, 80)

# Published settings used by the later sound and dead classifier. They are
# stored with every transform so later study replication scripts can verify
# that the correct volume version was used.
RADIAL_SUBBLOCK_WIDTH = 11
COARSE_RADIAL_STEP = 6
REPORTED_EVALUATIONS_PER_KNOT = 23

# Keep the pith placement used by the previous thesis replication.
RADIAL_PITH_OUTPUT_INDEX = 2
TANGENTIAL_AXIS_OUTPUT_INDEX = TARGET_BLOCK_SHAPE[1] // 2

# Thesis-style margins. Zero retains the previous replication behaviour.
RADIAL_INNER_MARGIN_MM = 0.0
RADIAL_OUTER_MARGIN_MM = 0.0
TANGENTIAL_MARGIN_MM = 0.0
LONGITUDINAL_MARGIN_SLICES = 0

# The direction estimate uses the outermost fraction of knot voxels relative
# to the pith. This reduces the influence of the wide knot root.
ORIENTATION_OUTER_FRACTION = 0.25

# Mask occupancy rules used during the first conversion from 0.5 mm to the
# coarse industry grid and again during local alignment.
MINIMUM_KNOT_OCCUPANCY = 0.25
MINIMUM_PITH_OCCUPANCY = 0.25
PRESERVE_LIVE_IF_MISSING = True
MINIMUM_LIVE_VOXELS_AFTER_RESAMPLING = 1
REQUIRE_LIVE_KNOT_AFTER_RESAMPLING = True
REQUIRE_PITH_AFTER_RESAMPLING = True

# Label fallback. It activates only when ordinary inverse sampling cannot
# retain the required number of live voxels. A radius of one permits a mapped
# live voxel that collides with pith to use its nearest non-pith neighbour.
ENABLE_GEOMETRIC_FORWARD_RESTORATION = True
FORWARD_RESTORATION_NEIGHBOUR_RADIUS = 1

INDEX_ORDER = "F"
COMPRESSION_LEVEL = 6
OVERWRITE_EXISTING_OUTPUT = False
VERIFY_WRITTEN_FILES = True
REFUSE_EXISTING_OUTPUT_ROOT = True


# -----------------------------------------------------------------------------
# Mask labels
# -----------------------------------------------------------------------------

BACKGROUND_LABEL = 0
LIVE_LABEL_MIN = 1
LIVE_LABEL_MAX = 50
DEAD_LABEL_OFFSET = 50
PITH_LABEL = 150
CLEAR_WOOD_LABEL = 255


HEADER_FIELDS_TO_REBUILD = {
    "type",
    "endian",
    "dimension",
    "sizes",
    "encoding",
    "data file",
    "datafile",
    "line skip",
    "byte skip",
    "block size",
    "old min",
    "old max",
    "oldmin",
    "oldmax",
    "min",
    "max",
    "spacings",
    "thicknesses",
    "sample units",
    "axis mins",
    "axis maxs",
    "axismins",
    "axismaxs",
}


@dataclass(frozen=True)
class KnotCropFiles:
    tree_number: int
    disk_number: int
    knot_id: int
    dry_image: str
    wet_image: str
    shared_mask: str


@dataclass(frozen=True)
class IndustryGrid:
    source_shape: Tuple[int, int, int]
    output_shape: Tuple[int, int, int]
    source_spacing: Tuple[float, float, float]
    target_spacing: Tuple[float, float, float]
    source_directions: np.ndarray
    target_directions: np.ndarray
    source_origin: np.ndarray
    target_origin: np.ndarray
    output_to_input_scale: np.ndarray
    output_to_input_offset: np.ndarray
    overlap_weights: Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LocalBlockTransform:
    angle_radians: float
    radial_unit_index: np.ndarray
    tangential_unit_index: np.ndarray
    pith_reference_industry_index: np.ndarray
    raw_local_knot_bounds: Tuple[float, float, float, float, float, float]
    local_knot_bounds: Tuple[float, float, float, float, float, float]
    required_native_shape: Tuple[int, int, int]
    scale_factors: Tuple[float, float, float]
    canonical_spacing_mm: Tuple[float, float, float]
    effective_spacing_mm: Tuple[float, float, float]
    padding_before: Tuple[int, int, int]
    padding_after: Tuple[int, int, int]
    output_to_industry_matrix: np.ndarray
    output_to_industry_offset: np.ndarray
    output_to_source_matrix: np.ndarray
    output_to_source_offset: np.ndarray
    target_directions: np.ndarray
    target_origin: np.ndarray


# -----------------------------------------------------------------------------
# Discovery and input validation
# -----------------------------------------------------------------------------


def validate_settings() -> None:
    if FIRST_TREE_NUMBER > LAST_TREE_NUMBER:
        raise ValueError(
            "FIRST_TREE_NUMBER cannot be greater than LAST_TREE_NUMBER."
        )

    if STEM_AXIS != 2:
        raise ValueError(
            "Local knot alignment in this script requires STEM_AXIS = 2."
        )

    if len(TARGET_BLOCK_SHAPE) != 3 or any(
        int(value) <= 1 for value in TARGET_BLOCK_SHAPE
    ):
        raise ValueError(
            "TARGET_BLOCK_SHAPE must contain three values greater than one."
        )

    radial_size, tangential_size, _ = TARGET_BLOCK_SHAPE
    if not 0 <= RADIAL_PITH_OUTPUT_INDEX < radial_size:
        raise ValueError("RADIAL_PITH_OUTPUT_INDEX is outside the block.")
    if not 0 <= TANGENTIAL_AXIS_OUTPUT_INDEX < tangential_size:
        raise ValueError(
            "TANGENTIAL_AXIS_OUTPUT_INDEX is outside the block."
        )

    if RADIAL_SUBBLOCK_WIDTH <= 0 or (
        RADIAL_SUBBLOCK_WIDTH % 2 == 0
    ):
        raise ValueError(
            "RADIAL_SUBBLOCK_WIDTH must be a positive odd value."
        )
    if RADIAL_SUBBLOCK_WIDTH > radial_size:
        raise ValueError(
            "RADIAL_SUBBLOCK_WIDTH cannot exceed the radial block size."
        )
    if COARSE_RADIAL_STEP <= 0:
        raise ValueError("COARSE_RADIAL_STEP must be positive.")

    for name, value in (
        ("RADIAL_INNER_MARGIN_MM", RADIAL_INNER_MARGIN_MM),
        ("RADIAL_OUTER_MARGIN_MM", RADIAL_OUTER_MARGIN_MM),
        ("TANGENTIAL_MARGIN_MM", TANGENTIAL_MARGIN_MM),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} cannot be negative.")

    if not isinstance(LONGITUDINAL_MARGIN_SLICES, int) or (
        LONGITUDINAL_MARGIN_SLICES < 0
    ):
        raise ValueError(
            "LONGITUDINAL_MARGIN_SLICES must be a nonnegative integer."
        )

    if REFUSE_EXISTING_OUTPUT_ROOT and OVERWRITE_EXISTING_OUTPUT:
        raise ValueError(
            "REFUSE_EXISTING_OUTPUT_ROOT and OVERWRITE_EXISTING_OUTPUT "
            "cannot both be true."
        )

    if MINIMUM_LIVE_VOXELS_AFTER_RESAMPLING < 0:
        raise ValueError(
            "MINIMUM_LIVE_VOXELS_AFTER_RESAMPLING cannot be negative."
        )
    if not isinstance(FORWARD_RESTORATION_NEIGHBOUR_RADIUS, int) or (
        FORWARD_RESTORATION_NEIGHBOUR_RADIUS < 0
    ):
        raise ValueError(
            "FORWARD_RESTORATION_NEIGHBOUR_RADIUS must be a nonnegative "
            "integer."
        )

    if not 0.0 < ORIENTATION_OUTER_FRACTION <= 1.0:
        raise ValueError(
            "ORIENTATION_OUTER_FRACTION must be greater than zero and at "
            "most one."
        )

    for name, value in (
        ("MINIMUM_KNOT_OCCUPANCY", MINIMUM_KNOT_OCCUPANCY),
        ("MINIMUM_PITH_OCCUPANCY", MINIMUM_PITH_OCCUPANCY),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie between zero and one.")


def numbered_subdirectories(parent: str, prefix: str):
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    matches = []

    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        match = pattern.fullmatch(name)
        if match and os.path.isdir(path):
            matches.append((int(match.group(1)), path))

    return sorted(matches, key=lambda item: item[0])


def find_knot_crops(
    source_root: str,
    tree_number: int,
) -> List[KnotCropFiles]:
    tree_directory = os.path.join(source_root, f"Tree{tree_number:02d}")
    if not os.path.isdir(tree_directory):
        raise FileNotFoundError(
            f"Source tree folder does not exist: {tree_directory}"
        )

    samples: List[KnotCropFiles] = []
    for disk_number, disk_directory in numbered_subdirectories(
        tree_directory,
        "Disk",
    ):
        for knot_id, knot_directory in numbered_subdirectories(
            disk_directory,
            "Knot",
        ):
            prefix = (
                f"Tree{tree_number:02d}_Disk{disk_number:02d}_"
                f"Knot{knot_id:02d}"
            )
            sample = KnotCropFiles(
                tree_number=tree_number,
                disk_number=disk_number,
                knot_id=knot_id,
                dry_image=os.path.join(
                    knot_directory,
                    f"Dry_{prefix}.nhdr",
                ),
                wet_image=os.path.join(
                    knot_directory,
                    f"Wet_{prefix}.nhdr",
                ),
                shared_mask=os.path.join(
                    knot_directory,
                    f"Mask_{prefix}.nhdr",
                ),
            )

            required_paths = (
                sample.dry_image,
                sample.dry_image[:-5] + ".raw.gz",
                sample.wet_image,
                sample.wet_image[:-5] + ".raw.gz",
                sample.shared_mask,
                sample.shared_mask[:-5] + ".raw.gz",
            )
            missing = [
                path for path in required_paths if not os.path.isfile(path)
            ]
            if missing:
                print(
                    f"Skipping Tree {tree_number:02d}, "
                    f"Disk {disk_number:02d}, Knot {knot_id:02d}."
                )
                for path in missing:
                    print(f"  Missing: {path}")
                continue

            samples.append(sample)

    return samples


def read_nhdr(path: str):
    if not path.lower().endswith(".nhdr"):
        raise ValueError(f"Only NHDR input is allowed: {path}")
    return nrrd.read(path, index_order=INDEX_ORDER)


def geometry_from_header(header: dict, name: str):
    if "space directions" not in header:
        raise ValueError(f"{name} has no space directions field.")
    if "space origin" not in header:
        raise ValueError(f"{name} has no space origin field.")

    try:
        directions = np.asarray(header["space directions"], dtype=float)
        origin = np.asarray(header["space origin"], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} has invalid physical geometry.") from error

    if directions.shape != (3, 3):
        raise ValueError(
            f"{name} space directions must have shape 3 by 3, found "
            f"{directions.shape}."
        )
    if origin.shape != (3,):
        raise ValueError(
            f"{name} space origin must contain three values, found "
            f"{origin.shape}."
        )
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(origin)):
        raise ValueError(f"{name} contains nonfinite geometry values.")

    spacing = np.linalg.norm(directions, axis=1)
    if np.any(spacing <= 0):
        raise ValueError(f"{name} contains a zero length axis direction.")

    return directions, spacing, origin


def geometry_matches(
    reference_header: dict,
    other_header: dict,
    reference_name: str,
    other_name: str,
) -> None:
    ref_directions, ref_spacing, ref_origin = geometry_from_header(
        reference_header,
        reference_name,
    )
    other_directions, other_spacing, other_origin = geometry_from_header(
        other_header,
        other_name,
    )

    if not np.allclose(ref_spacing, other_spacing, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{reference_name} and {other_name} have different spacing."
        )
    if not np.allclose(
        ref_directions,
        other_directions,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{reference_name} and {other_name} have different directions."
        )
    if not np.allclose(ref_origin, other_origin, rtol=0.0, atol=1e-5):
        raise ValueError(
            f"{reference_name} and {other_name} have different origins."
        )


def validate_input_set(
    sample: KnotCropFiles,
    dry_data: np.ndarray,
    dry_header: dict,
    wet_data: np.ndarray,
    wet_header: dict,
    mask_data: np.ndarray,
    mask_header: dict,
) -> None:
    arrays = {
        "dry image": dry_data,
        "wet image": wet_data,
        "shared mask": mask_data,
    }
    if any(data.ndim != 3 for data in arrays.values()):
        details = ", ".join(
            f"{name}={data.shape}" for name, data in arrays.items()
        )
        raise ValueError(f"Input is not three dimensional: {details}")

    shapes = {data.shape for data in arrays.values()}
    if len(shapes) != 1:
        details = ", ".join(
            f"{name}={data.shape}" for name, data in arrays.items()
        )
        raise ValueError(
            f"Tree {sample.tree_number:02d}, Disk {sample.disk_number:02d}, "
            f"Knot {sample.knot_id:02d} has mismatched shapes: {details}"
        )

    geometry_matches(
        dry_header,
        wet_header,
        "Dry source image",
        "Wet source image",
    )
    geometry_matches(
        dry_header,
        mask_header,
        "Dry source image",
        "Shared source mask",
    )


def integer_labels(data: np.ndarray, name: str) -> List[int]:
    if np.issubdtype(data.dtype, np.floating):
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{name} contains nonfinite values.")
        if not np.all(data == np.round(data)):
            raise ValueError(f"{name} contains noninteger labels.")
    return sorted(np.unique(data).astype(int).tolist())


def validate_source_mask(mask: np.ndarray, knot_id: int) -> List[int]:
    if not LIVE_LABEL_MIN <= knot_id <= LIVE_LABEL_MAX:
        raise ValueError(f"Knot ID {knot_id} is outside the range 1 to 50.")

    labels = integer_labels(mask, "Source shared mask")
    allowed = {
        BACKGROUND_LABEL,
        CLEAR_WOOD_LABEL,
        PITH_LABEL,
        knot_id,
        knot_id + DEAD_LABEL_OFFSET,
    }
    unexpected = [label for label in labels if label not in allowed]
    if unexpected:
        raise ValueError(
            f"Source mask contains unexpected labels for Knot {knot_id}: "
            f"{unexpected}"
        )
    if knot_id not in labels:
        raise ValueError(f"Source mask is missing live label {knot_id}.")
    if PITH_LABEL not in labels:
        raise ValueError(f"Source mask is missing pith label {PITH_LABEL}.")
    return labels


# -----------------------------------------------------------------------------
# First stage, physical conversion from 0.5 mm to 1 by 1 by 10 mm
# -----------------------------------------------------------------------------


def calculate_axis_overlap_weights(
    source_count: int,
    source_spacing: float,
    output_count: int,
    target_spacing: float,
    output_to_input_offset: float,
) -> np.ndarray:
    source_centres = np.arange(source_count, dtype=float) * source_spacing
    source_lower = source_centres - 0.5 * source_spacing
    source_upper = source_centres + 0.5 * source_spacing

    output_centres = (
        output_to_input_offset
        + np.arange(output_count, dtype=float)
        * target_spacing
        / source_spacing
    ) * source_spacing
    output_lower = output_centres - 0.5 * target_spacing
    output_upper = output_centres + 0.5 * target_spacing

    overlap = np.maximum(
        0.0,
        np.minimum(output_upper[:, None], source_upper[None, :])
        - np.maximum(output_lower[:, None], source_lower[None, :]),
    )
    covered_length = overlap.sum(axis=1, keepdims=True)
    if np.any(covered_length <= 0):
        raise ValueError(
            "At least one industry grid voxel has no overlap with the source."
        )

    overlap /= covered_length
    return overlap.astype(np.float32)


def calculate_industry_grid(
    source_shape: Sequence[int],
    source_header: dict,
) -> IndustryGrid:
    if len(source_shape) != 3:
        raise ValueError(f"Expected a three dimensional shape: {source_shape}")

    source_directions, source_spacing, source_origin = geometry_from_header(
        source_header,
        "Dry source image",
    )
    expected_spacing = np.asarray(EXPECTED_SOURCE_SPACING_MM, dtype=float)
    if not np.allclose(
        source_spacing,
        expected_spacing,
        rtol=0.0,
        atol=SPACING_TOLERANCE_MM,
    ):
        raise ValueError(
            f"Source spacing is {source_spacing.tolist()} mm, expected "
            f"{expected_spacing.tolist()} mm."
        )

    source_shape_array = np.asarray(source_shape, dtype=int)
    if np.any(source_shape_array <= 0):
        raise ValueError(f"Invalid source shape: {tuple(source_shape)}")

    target_spacing = np.asarray(INDUSTRY_SPACING_MM, dtype=float)
    unit_directions = source_directions / source_spacing[:, None]
    target_directions = unit_directions * target_spacing[:, None]

    floating_shape = (
        source_shape_array.astype(float)
        * source_spacing
        / target_spacing
    )
    output_shape_array = np.maximum(
        1,
        np.floor(floating_shape + 0.5).astype(int),
    )

    source_centre = source_origin.copy()
    target_centre_offset = np.zeros(3, dtype=float)
    for axis in range(3):
        source_centre += (
            0.5
            * (source_shape_array[axis] - 1)
            * source_directions[axis]
        )
        target_centre_offset += (
            0.5
            * (output_shape_array[axis] - 1)
            * target_directions[axis]
        )

    target_origin = source_centre - target_centre_offset
    output_to_input_scale = target_spacing / source_spacing
    source_index_centre = 0.5 * (source_shape_array.astype(float) - 1.0)
    output_index_centre = 0.5 * (output_shape_array.astype(float) - 1.0)
    output_to_input_offset = (
        source_index_centre
        - output_to_input_scale * output_index_centre
    )

    overlap_weights = tuple(
        calculate_axis_overlap_weights(
            source_count=int(source_shape_array[axis]),
            source_spacing=float(source_spacing[axis]),
            output_count=int(output_shape_array[axis]),
            target_spacing=float(target_spacing[axis]),
            output_to_input_offset=float(output_to_input_offset[axis]),
        )
        for axis in range(3)
    )

    return IndustryGrid(
        source_shape=tuple(int(value) for value in source_shape_array),
        output_shape=tuple(int(value) for value in output_shape_array),
        source_spacing=tuple(float(value) for value in source_spacing),
        target_spacing=tuple(float(value) for value in target_spacing),
        source_directions=source_directions,
        target_directions=target_directions,
        source_origin=source_origin,
        target_origin=target_origin,
        output_to_input_scale=output_to_input_scale,
        output_to_input_offset=output_to_input_offset,
        overlap_weights=overlap_weights,
    )


def physical_volume_average(data: np.ndarray, grid: IndustryGrid) -> np.ndarray:
    working = data.astype(np.float32, copy=False)
    weights_0, weights_1, weights_2 = grid.overlap_weights

    reduced = np.tensordot(weights_0, working, axes=(1, 0))
    reduced = np.tensordot(
        weights_1,
        reduced,
        axes=(1, 1),
    ).transpose(1, 0, 2)
    reduced = np.tensordot(
        weights_2,
        reduced,
        axes=(1, 2),
    ).transpose(1, 2, 0)

    if reduced.shape != grid.output_shape:
        raise ValueError(
            f"Volume averaging produced {reduced.shape}, expected "
            f"{grid.output_shape}."
        )
    return reduced


def cast_image_like(data: np.ndarray, source_dtype: np.dtype) -> np.ndarray:
    source_dtype = np.dtype(source_dtype)
    if np.issubdtype(source_dtype, np.integer):
        limits = np.iinfo(source_dtype)
        data = np.clip(np.rint(data), limits.min, limits.max)
    return data.astype(source_dtype, copy=False)


def resample_image_to_industry(
    data: np.ndarray,
    grid: IndustryGrid,
) -> np.ndarray:
    return cast_image_like(physical_volume_average(data, grid), data.dtype)


def preserve_pith_in_supported_slices(
    output_mask: np.ndarray,
    pith_support: np.ndarray,
) -> int:
    restored = 0
    for longitudinal_index in range(output_mask.shape[2]):
        score = pith_support[:, :, longitudinal_index]
        if not np.any(score > 0):
            continue
        output_slice = output_mask[:, :, longitudinal_index]
        if np.any(output_slice == PITH_LABEL):
            continue
        index_2d = np.unravel_index(int(np.argmax(score)), score.shape)
        output_mask[index_2d[0], index_2d[1], longitudinal_index] = PITH_LABEL
        restored += 1
    return restored


def forward_neighbour_offsets(radius: int) -> List[Tuple[int, int, int]]:
    offsets = [
        (axis_0, axis_1, axis_2)
        for axis_0 in range(-radius, radius + 1)
        for axis_1 in range(-radius, radius + 1)
        for axis_2 in range(-radius, radius + 1)
    ]
    return sorted(
        offsets,
        key=lambda value: (
            value[0] ** 2 + value[1] ** 2 + value[2] ** 2,
            value,
        ),
    )


def geometric_forward_live_candidates(
    output_mask: np.ndarray,
    source_mask: np.ndarray,
    source_to_output_matrix: np.ndarray,
    source_to_output_offset: np.ndarray,
    live_label: int,
    live_support: np.ndarray,
    knot_support: np.ndarray,
) -> Tuple[List[dict], dict]:
    """Map genuine source live voxels to nearby non-pith output voxels."""

    matrix = np.asarray(source_to_output_matrix, dtype=float)
    offset = np.asarray(source_to_output_offset, dtype=float)
    if matrix.shape != (3, 3) or offset.shape != (3,):
        raise ValueError("Forward restoration transform has an invalid shape.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
        raise ValueError("Forward restoration transform is not finite.")

    source_coordinates = np.argwhere(source_mask == live_label).astype(float)
    if source_coordinates.size == 0:
        raise ValueError(
            f"Source mask contains no voxels for live label {live_label}."
        )

    mapped_coordinates = (
        matrix @ source_coordinates.T
        + offset[:, None]
    ).T
    mapped_coordinates = mapped_coordinates[
        np.all(np.isfinite(mapped_coordinates), axis=1)
    ]
    output_shape = np.asarray(output_mask.shape, dtype=int)
    offsets = forward_neighbour_offsets(
        FORWARD_RESTORATION_NEIGHBOUR_RADIUS
    )

    votes: Dict[Tuple[int, int, int], dict] = {}
    base_in_bounds = 0
    base_pith_collisions = 0
    relocated_source_voxels = 0
    source_voxels_without_candidate = 0

    for coordinate in mapped_coordinates:
        rounded = np.rint(coordinate).astype(int)
        rounded_in_bounds = bool(
            np.all(rounded >= 0) and np.all(rounded < output_shape)
        )
        if rounded_in_bounds:
            base_in_bounds += 1
            rounded_tuple = tuple(int(value) for value in rounded)
            if output_mask[rounded_tuple] == PITH_LABEL:
                base_pith_collisions += 1

        selected = None
        selected_distance = math.inf
        for neighbour_offset in offsets:
            candidate = rounded + np.asarray(neighbour_offset, dtype=int)
            if np.any(candidate < 0) or np.any(candidate >= output_shape):
                continue
            candidate_tuple = tuple(int(value) for value in candidate)
            if output_mask[candidate_tuple] in (PITH_LABEL, live_label):
                continue

            squared_distance = float(
                np.sum((candidate.astype(float) - coordinate) ** 2)
            )
            if squared_distance < selected_distance - 1e-12:
                selected = candidate_tuple
                selected_distance = squared_distance

        if selected is None:
            source_voxels_without_candidate += 1
            continue

        rounded_tuple = tuple(int(value) for value in rounded)
        if selected != rounded_tuple:
            relocated_source_voxels += 1

        record = votes.setdefault(
            selected,
            {
                "index": selected,
                "source_votes": 0,
                "sum_squared_distance": 0.0,
                "minimum_squared_distance": math.inf,
            },
        )
        record["source_votes"] += 1
        record["sum_squared_distance"] += selected_distance
        record["minimum_squared_distance"] = min(
            record["minimum_squared_distance"],
            selected_distance,
        )

    candidates = []
    for index, record in votes.items():
        source_votes = int(record["source_votes"])
        candidates.append(
            {
                "index": index,
                "source_votes": source_votes,
                "mean_squared_distance": (
                    float(record["sum_squared_distance"]) / source_votes
                ),
                "minimum_squared_distance": float(
                    record["minimum_squared_distance"]
                ),
                "inverse_live_support": float(live_support[index]),
                "inverse_knot_support": float(knot_support[index]),
            }
        )

    candidates.sort(
        key=lambda record: (
            -int(
                record["inverse_knot_support"]
                >= MINIMUM_KNOT_OCCUPANCY
            ),
            -int(record["source_votes"]),
            -float(record["inverse_live_support"]),
            float(record["mean_squared_distance"]),
            tuple(record["index"]),
        )
    )
    diagnostics = {
        "source_live_voxels": int(source_coordinates.shape[0]),
        "finite_mapped_source_voxels": int(mapped_coordinates.shape[0]),
        "base_in_bounds_source_voxels": base_in_bounds,
        "base_pith_collisions": base_pith_collisions,
        "relocated_source_voxels": relocated_source_voxels,
        "source_voxels_without_candidate": source_voxels_without_candidate,
        "candidate_output_voxels": len(candidates),
        "neighbour_radius": FORWARD_RESTORATION_NEIGHBOUR_RADIUS,
    }
    return candidates, diagnostics


def preserve_live_from_support(
    output_mask: np.ndarray,
    live_support: np.ndarray,
    knot_support: np.ndarray,
    live_label: int,
    source_mask: np.ndarray,
    source_to_output_matrix: np.ndarray,
    source_to_output_offset: np.ndarray,
    stage_name: str,
) -> dict:
    required = max(0, int(MINIMUM_LIVE_VOXELS_AFTER_RESAMPLING))
    current = int(np.count_nonzero(output_mask == live_label))
    result = {
        "stage": stage_name,
        "applied": False,
        "method": "not_needed",
        "required_live_voxels": required,
        "live_voxels_before": current,
        "live_voxels_after": current,
        "restored_voxels": 0,
        "restored_indices": [],
        "restored_support": [],
        "restored_records": [],
        "forward_mapping": None,
    }

    while current < required:
        inverse_candidates = (
            (live_support > 0)
            & (output_mask != PITH_LABEL)
            & (output_mask != live_label)
        )
        if np.any(inverse_candidates):
            preferred = inverse_candidates & (
                knot_support >= MINIMUM_KNOT_OCCUPANCY
            )
            eligible = preferred if np.any(preferred) else inverse_candidates
            score = np.where(eligible, live_support, -np.inf)
            radial_penalty = (
                np.arange(output_mask.shape[0], dtype=float)[:, None, None]
                * 1e-9
            )
            selected_flat = int(np.argmax(score - radial_penalty))
            selected = tuple(
                int(value)
                for value in np.unravel_index(
                    selected_flat,
                    output_mask.shape,
                )
            )
            method = "inverse_sample_support"
            record = {
                "index": selected,
                "method": method,
                "inverse_live_support": float(live_support[selected]),
                "inverse_knot_support": float(knot_support[selected]),
            }
        else:
            if not ENABLE_GEOMETRIC_FORWARD_RESTORATION:
                raise ValueError(
                    f"{stage_name}: live label {live_label} vanished and "
                    "geometric forward restoration is disabled."
                )

            forward_candidates, diagnostics = (
                geometric_forward_live_candidates(
                    output_mask=output_mask,
                    source_mask=source_mask,
                    source_to_output_matrix=source_to_output_matrix,
                    source_to_output_offset=source_to_output_offset,
                    live_label=live_label,
                    live_support=live_support,
                    knot_support=knot_support,
                )
            )
            result["forward_mapping"] = diagnostics
            if not forward_candidates:
                raise ValueError(
                    f"{stage_name}: live label {live_label} vanished. "
                    "Forward mapping found no valid non-pith output voxel. "
                    f"Diagnostics: {diagnostics}"
                )
            record = forward_candidates[0]
            selected = tuple(int(value) for value in record["index"])
            method = "geometric_forward_mapping"
            record = dict(record)
            record["method"] = method

        output_mask[selected] = live_label
        result["applied"] = True
        if result["method"] == "not_needed":
            result["method"] = method
        elif result["method"] != method:
            result["method"] = "inverse_then_geometric_forward_mapping"
        result["restored_voxels"] += 1
        result["restored_indices"].append(selected)
        result["restored_support"].append(float(live_support[selected]))
        result["restored_records"].append(record)
        current = int(np.count_nonzero(output_mask == live_label))

    result["live_voxels_after"] = current
    return result


def resample_mask_to_industry(
    data: np.ndarray,
    grid: IndustryGrid,
    knot_id: int,
) -> Tuple[np.ndarray, dict]:
    live_label = knot_id
    dead_label = knot_id + DEAD_LABEL_OFFSET

    background = physical_volume_average(data == BACKGROUND_LABEL, grid)
    clear_wood = physical_volume_average(data == CLEAR_WOOD_LABEL, grid)
    live = physical_volume_average(data == live_label, grid)
    dead = physical_volume_average(data == dead_label, grid)
    pith = physical_volume_average(data == PITH_LABEL, grid)

    occupancy_sum = background + clear_wood + live + dead + pith
    if not np.allclose(occupancy_sum, 1.0, rtol=0.0, atol=2e-5):
        raise ValueError(
            "Industry mask occupancies do not sum to one in every voxel."
        )

    output = np.where(
        background > clear_wood,
        BACKGROUND_LABEL,
        CLEAR_WOOD_LABEL,
    ).astype(data.dtype)
    combined_knot = live + dead
    knot_present = combined_knot >= MINIMUM_KNOT_OCCUPANCY
    live_dominant = live >= dead
    output[knot_present & live_dominant] = live_label
    output[knot_present & ~live_dominant] = dead_label
    output[pith >= MINIMUM_PITH_OCCUPANCY] = PITH_LABEL

    restored_pith_slices = preserve_pith_in_supported_slices(output, pith)
    current_live = int(np.count_nonzero(output == live_label))
    live_preservation = {
        "stage": "industry_grid_1_by_1_by_10_mm",
        "applied": False,
        "method": "disabled",
        "required_live_voxels": int(
            MINIMUM_LIVE_VOXELS_AFTER_RESAMPLING
        ),
        "live_voxels_before": current_live,
        "live_voxels_after": current_live,
        "restored_voxels": 0,
        "restored_indices": [],
        "restored_support": [],
        "restored_records": [],
        "forward_mapping": None,
    }
    if PRESERVE_LIVE_IF_MISSING:
        source_to_output_scale = 1.0 / grid.output_to_input_scale
        source_to_output_matrix = np.diag(source_to_output_scale)
        source_to_output_offset = (
            -grid.output_to_input_offset * source_to_output_scale
        )
        live_preservation = preserve_live_from_support(
            output,
            live,
            combined_knot,
            live_label,
            source_mask=data,
            source_to_output_matrix=source_to_output_matrix,
            source_to_output_offset=source_to_output_offset,
            stage_name="industry_grid_1_by_1_by_10_mm",
        )

    summary = {
        "restored_pith_slices": restored_pith_slices,
        "live_preservation": live_preservation,
        "live_voxels": int(np.count_nonzero(output == live_label)),
        "dead_voxels": int(np.count_nonzero(output == dead_label)),
        "pith_voxels": int(np.count_nonzero(output == PITH_LABEL)),
    }
    return output, summary


# -----------------------------------------------------------------------------
# Local radial, tangential, and longitudinal coordinate construction
# -----------------------------------------------------------------------------


def pith_centres_by_longitudinal_slice(mask: np.ndarray) -> np.ndarray:
    centres = np.full((mask.shape[2], 2), np.nan, dtype=float)
    for longitudinal_index in range(mask.shape[2]):
        coordinates = np.argwhere(
            mask[:, :, longitudinal_index] == PITH_LABEL
        )
        if coordinates.size:
            centres[longitudinal_index] = coordinates.mean(axis=0)

    valid = np.flatnonzero(np.all(np.isfinite(centres), axis=1))
    if valid.size == 0:
        raise ValueError("Industry mask has no pith voxels.")

    all_indices = np.arange(mask.shape[2], dtype=float)
    for plane_axis in range(2):
        centres[:, plane_axis] = np.interp(
            all_indices,
            valid.astype(float),
            centres[valid, plane_axis],
        )
    return centres


def estimate_radial_direction(
    mask: np.ndarray,
    knot_id: int,
    pith_centres: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    knot_binary = (
        (mask == knot_id)
        | (mask == knot_id + DEAD_LABEL_OFFSET)
    )
    knot_coordinates = np.argwhere(knot_binary)
    if knot_coordinates.size == 0:
        raise ValueError("Industry mask has no knot voxels.")

    corresponding_pith = pith_centres[knot_coordinates[:, 2]]
    vectors = knot_coordinates[:, :2].astype(float) - corresponding_pith
    distances = np.linalg.norm(vectors, axis=1)
    valid = distances > 1e-6
    if not np.any(valid):
        raise ValueError("Knot voxels do not extend away from the pith.")

    vectors = vectors[valid]
    distances = distances[valid]
    cutoff = float(
        np.quantile(distances, 1.0 - ORIENTATION_OUTER_FRACTION)
    )
    selected = vectors[distances >= cutoff]
    mean_vector = selected.mean(axis=0)

    if np.linalg.norm(mean_vector) <= 1e-6:
        covariance = selected.T @ selected
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        mean_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
        all_mean = vectors.mean(axis=0)
        if float(np.dot(mean_vector, all_mean)) < 0:
            mean_vector = -mean_vector

    radial = mean_vector / np.linalg.norm(mean_vector)
    tangential = np.array([-radial[1], radial[0]], dtype=float)
    return radial, tangential


def required_padding(
    output_coordinates: np.ndarray,
    required_minimum: float,
    required_maximum: float,
) -> Tuple[int, int]:
    inside = (
        (output_coordinates >= required_minimum - 1e-6)
        & (output_coordinates <= required_maximum + 1e-6)
    )
    occupied = np.flatnonzero(inside)
    if occupied.size == 0:
        return 0, 0
    return int(occupied[0]), int(len(output_coordinates) - 1 - occupied[-1])


def build_local_transform(
    industry_mask: np.ndarray,
    industry_grid: IndustryGrid,
    knot_id: int,
) -> LocalBlockTransform:
    radial_size, tangential_size, longitudinal_size = TARGET_BLOCK_SHAPE
    pith_centres = pith_centres_by_longitudinal_slice(industry_mask)
    radial, tangential = estimate_radial_direction(
        industry_mask,
        knot_id,
        pith_centres,
    )

    knot_binary = (
        (industry_mask == knot_id)
        | (industry_mask == knot_id + DEAD_LABEL_OFFSET)
    )
    knot_coordinates = np.argwhere(knot_binary).astype(float)
    raw_longitudinal_minimum = int(np.min(knot_coordinates[:, 2]))
    raw_longitudinal_maximum = int(np.max(knot_coordinates[:, 2]))

    reference_slice_indices = np.arange(
        raw_longitudinal_minimum,
        raw_longitudinal_maximum + 1,
        dtype=int,
    )
    pith_reference_xy = np.median(
        pith_centres[reference_slice_indices],
        axis=0,
    )
    longitudinal_reference = 0.5 * (
        raw_longitudinal_minimum + raw_longitudinal_maximum
    )
    pith_reference = np.array(
        [
            float(pith_reference_xy[0]),
            float(pith_reference_xy[1]),
            float(longitudinal_reference),
        ],
        dtype=float,
    )

    relative_xy = knot_coordinates[:, :2] - pith_reference_xy
    radial_values = relative_xy @ radial
    tangential_values = relative_xy @ tangential

    # Ensure the selected radial direction points toward most of the knot.
    if float(np.mean(radial_values)) < 0:
        radial = -radial
        tangential = -tangential
        radial_values = -radial_values
        tangential_values = -tangential_values

    raw_radial_minimum = min(float(np.min(radial_values)), 0.0)
    raw_radial_maximum = max(float(np.max(radial_values)), 0.0)
    raw_tangential_minimum = float(np.min(tangential_values))
    raw_tangential_maximum = float(np.max(tangential_values))

    in_plane_spacing = float(INDUSTRY_SPACING_MM[0])
    inner_margin_index = RADIAL_INNER_MARGIN_MM / in_plane_spacing
    outer_margin_index = RADIAL_OUTER_MARGIN_MM / in_plane_spacing
    tangential_margin_index = TANGENTIAL_MARGIN_MM / in_plane_spacing

    radial_minimum = raw_radial_minimum - inner_margin_index
    radial_maximum = raw_radial_maximum + outer_margin_index
    tangential_minimum = (
        raw_tangential_minimum - tangential_margin_index
    )
    tangential_maximum = (
        raw_tangential_maximum + tangential_margin_index
    )
    longitudinal_minimum = max(
        0,
        raw_longitudinal_minimum - LONGITUDINAL_MARGIN_SLICES,
    )
    longitudinal_maximum = min(
        industry_mask.shape[2] - 1,
        raw_longitudinal_maximum + LONGITUDINAL_MARGIN_SLICES,
    )

    radial_negative_capacity = RADIAL_PITH_OUTPUT_INDEX
    radial_positive_capacity = (
        radial_size - 1 - RADIAL_PITH_OUTPUT_INDEX
    )
    # Starting with 1.0 is the key thesis fitting rule. It prevents a knot
    # that already fits from being enlarged to fill the output block.
    radial_scale_candidates = [1.0]
    if radial_negative_capacity > 0:
        radial_scale_candidates.append(
            max(0.0, -radial_minimum) / radial_negative_capacity
        )
    elif radial_minimum < -1e-6:
        raise ValueError(
            "The knot extends behind the pith, but the configured radial "
            "pith index leaves no negative radial capacity."
        )
    if radial_positive_capacity > 0:
        radial_scale_candidates.append(
            max(0.0, radial_maximum) / radial_positive_capacity
        )
    radial_scale = max(radial_scale_candidates)

    tangential_negative_capacity = TANGENTIAL_AXIS_OUTPUT_INDEX
    tangential_positive_capacity = (
        tangential_size - 1 - TANGENTIAL_AXIS_OUTPUT_INDEX
    )
    tangential_scale_candidates = [1.0]
    if tangential_negative_capacity > 0:
        tangential_scale_candidates.append(
            max(0.0, -tangential_minimum)
            / tangential_negative_capacity
        )
    if tangential_positive_capacity > 0:
        tangential_scale_candidates.append(
            max(0.0, tangential_maximum)
            / tangential_positive_capacity
        )
    tangential_scale = max(tangential_scale_candidates)

    required_longitudinal_count = (
        longitudinal_maximum - longitudinal_minimum + 1
    )
    if required_longitudinal_count <= longitudinal_size:
        longitudinal_scale = 1.0
        extra = longitudinal_size - required_longitudinal_count
        longitudinal_start = longitudinal_minimum - extra // 2
    else:
        longitudinal_scale = (
            longitudinal_maximum - longitudinal_minimum
        ) / (longitudinal_size - 1)
        longitudinal_start = float(longitudinal_minimum)

    if min(radial_scale, tangential_scale, longitudinal_scale) < 1.0:
        raise AssertionError(
            "Thesis fitting produced an enlargement scale below one."
        )

    radial_coordinates = (
        np.arange(radial_size, dtype=float)
        - float(RADIAL_PITH_OUTPUT_INDEX)
    ) * radial_scale
    tangential_coordinates = (
        np.arange(tangential_size, dtype=float)
        - float(TANGENTIAL_AXIS_OUTPUT_INDEX)
    ) * tangential_scale
    longitudinal_coordinates = (
        float(longitudinal_start)
        + np.arange(longitudinal_size, dtype=float)
        * longitudinal_scale
    )

    required_radial_count = int(
        math.ceil(radial_maximum) - math.floor(radial_minimum) + 1
    )
    required_tangential_count = int(
        math.ceil(tangential_maximum)
        - math.floor(tangential_minimum)
        + 1
    )
    required_longitudinal_count = int(required_longitudinal_count)

    radial_padding = required_padding(
        radial_coordinates,
        radial_minimum,
        radial_maximum,
    )
    tangential_padding = required_padding(
        tangential_coordinates,
        tangential_minimum,
        tangential_maximum,
    )
    longitudinal_padding = required_padding(
        longitudinal_coordinates,
        float(longitudinal_minimum),
        float(longitudinal_maximum),
    )

    output_to_industry_matrix = np.array(
        [
            [
                radial[0] * radial_scale,
                tangential[0] * tangential_scale,
                0.0,
            ],
            [
                radial[1] * radial_scale,
                tangential[1] * tangential_scale,
                0.0,
            ],
            [0.0, 0.0, longitudinal_scale],
        ],
        dtype=float,
    )

    radial_zero_coordinate = (
        -float(RADIAL_PITH_OUTPUT_INDEX) * radial_scale
    )
    tangential_zero_coordinate = (
        -float(TANGENTIAL_AXIS_OUTPUT_INDEX) * tangential_scale
    )
    output_to_industry_offset = np.array(
        [
            pith_reference_xy[0]
            + radial[0] * radial_zero_coordinate
            + tangential[0] * tangential_zero_coordinate,
            pith_reference_xy[1]
            + radial[1] * radial_zero_coordinate
            + tangential[1] * tangential_zero_coordinate,
            float(longitudinal_start),
        ],
        dtype=float,
    )

    source_scale_matrix = np.diag(
        industry_grid.output_to_input_scale.astype(float)
    )
    output_to_source_matrix = (
        source_scale_matrix @ output_to_industry_matrix
    )
    output_to_source_offset = (
        industry_grid.output_to_input_scale
        * output_to_industry_offset
        + industry_grid.output_to_input_offset
    )

    target_directions = np.vstack(
        [
            output_to_industry_matrix[:, output_axis]
            @ industry_grid.target_directions
            for output_axis in range(3)
        ]
    )
    target_origin = (
        industry_grid.target_origin
        + output_to_industry_offset @ industry_grid.target_directions
    )
    effective_spacing = np.linalg.norm(target_directions, axis=1)

    angle_radians = math.atan2(float(radial[1]), float(radial[0]))
    return LocalBlockTransform(
        angle_radians=angle_radians,
        radial_unit_index=radial,
        tangential_unit_index=tangential,
        pith_reference_industry_index=pith_reference,
        raw_local_knot_bounds=(
            raw_radial_minimum,
            raw_radial_maximum,
            raw_tangential_minimum,
            raw_tangential_maximum,
            float(raw_longitudinal_minimum),
            float(raw_longitudinal_maximum),
        ),
        local_knot_bounds=(
            radial_minimum,
            radial_maximum,
            tangential_minimum,
            tangential_maximum,
            float(longitudinal_minimum),
            float(longitudinal_maximum),
        ),
        required_native_shape=(
            required_radial_count,
            required_tangential_count,
            required_longitudinal_count,
        ),
        scale_factors=(
            float(radial_scale),
            float(tangential_scale),
            float(longitudinal_scale),
        ),
        canonical_spacing_mm=tuple(float(v) for v in INDUSTRY_SPACING_MM),
        effective_spacing_mm=tuple(float(v) for v in effective_spacing),
        padding_before=(
            radial_padding[0],
            tangential_padding[0],
            longitudinal_padding[0],
        ),
        padding_after=(
            radial_padding[1],
            tangential_padding[1],
            longitudinal_padding[1],
        ),
        output_to_industry_matrix=output_to_industry_matrix,
        output_to_industry_offset=output_to_industry_offset,
        output_to_source_matrix=output_to_source_matrix,
        output_to_source_offset=output_to_source_offset,
        target_directions=target_directions,
        target_origin=target_origin,
    )


def source_coordinates_for_block(
    transform: LocalBlockTransform,
) -> np.ndarray:
    output_indices = np.indices(TARGET_BLOCK_SHAPE, dtype=np.float64)
    flat_output = output_indices.reshape(3, -1)
    flat_source = (
        transform.output_to_industry_matrix @ flat_output
        + transform.output_to_industry_offset[:, None]
    )
    return flat_source.reshape((3, *TARGET_BLOCK_SHAPE))


# -----------------------------------------------------------------------------
# Final local resampling
# -----------------------------------------------------------------------------


def background_fill_value(
    image: np.ndarray,
    mask: np.ndarray,
) -> float:
    values = image[mask == BACKGROUND_LABEL]
    if values.size:
        return float(np.median(values))
    return float(np.min(image))


def resample_local_image(
    image: np.ndarray,
    coordinates: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    sampled = map_coordinates(
        image.astype(np.float32, copy=False),
        coordinates,
        order=1,
        mode="constant",
        cval=float(fill_value),
        prefilter=False,
    )
    return cast_image_like(sampled, image.dtype)


def sample_class_support(
    mask: np.ndarray,
    label: int,
    coordinates: np.ndarray,
    outside_value: float,
    order: int,
) -> np.ndarray:
    binary = (mask == label).astype(np.float32)
    return map_coordinates(
        binary,
        coordinates,
        order=order,
        mode="constant",
        cval=float(outside_value),
        prefilter=False,
    )


def resample_mask_to_local_block(
    mask: np.ndarray,
    coordinates: np.ndarray,
    transform: LocalBlockTransform,
    knot_id: int,
) -> Tuple[np.ndarray, dict]:
    live_label = knot_id
    dead_label = knot_id + DEAD_LABEL_OFFSET

    background = sample_class_support(
        mask,
        BACKGROUND_LABEL,
        coordinates,
        outside_value=1.0,
        order=1,
    )
    clear_wood = sample_class_support(
        mask,
        CLEAR_WOOD_LABEL,
        coordinates,
        outside_value=0.0,
        order=1,
    )
    live = sample_class_support(
        mask,
        live_label,
        coordinates,
        outside_value=0.0,
        order=1,
    )
    dead = sample_class_support(
        mask,
        dead_label,
        coordinates,
        outside_value=0.0,
        order=1,
    )
    pith = sample_class_support(
        mask,
        PITH_LABEL,
        coordinates,
        outside_value=0.0,
        order=1,
    )

    occupancy_sum = background + clear_wood + live + dead + pith
    if not np.allclose(occupancy_sum, 1.0, rtol=0.0, atol=3e-4):
        maximum_error = float(np.max(np.abs(occupancy_sum - 1.0)))
        raise ValueError(
            "Local mask occupancies do not sum to one. Maximum error is "
            f"{maximum_error:.6g}."
        )

    output = np.where(
        background > clear_wood,
        BACKGROUND_LABEL,
        CLEAR_WOOD_LABEL,
    ).astype(mask.dtype)
    combined_knot = live + dead
    knot_present = combined_knot >= MINIMUM_KNOT_OCCUPANCY
    live_dominant = live >= dead
    output[knot_present & live_dominant] = live_label
    output[knot_present & ~live_dominant] = dead_label

    pith_nearest = sample_class_support(
        mask,
        PITH_LABEL,
        coordinates,
        outside_value=0.0,
        order=0,
    )
    pith_support = np.maximum(pith, pith_nearest)
    output[pith >= MINIMUM_PITH_OCCUPANCY] = PITH_LABEL
    restored_pith_slices = preserve_pith_in_supported_slices(
        output,
        pith_support,
    )

    current_live = int(np.count_nonzero(output == live_label))
    live_preservation = {
        "stage": "thesis_fit_block_160_by_80_by_80",
        "applied": False,
        "method": "disabled",
        "required_live_voxels": int(
            MINIMUM_LIVE_VOXELS_AFTER_RESAMPLING
        ),
        "live_voxels_before": current_live,
        "live_voxels_after": current_live,
        "restored_voxels": 0,
        "restored_indices": [],
        "restored_support": [],
        "restored_records": [],
        "forward_mapping": None,
    }
    if PRESERVE_LIVE_IF_MISSING:
        live_nearest = sample_class_support(
            mask,
            live_label,
            coordinates,
            outside_value=0.0,
            order=0,
        )
        live_support = np.maximum(live, live_nearest)
        industry_to_output_matrix = np.linalg.inv(
            transform.output_to_industry_matrix
        )
        industry_to_output_offset = (
            -industry_to_output_matrix
            @ transform.output_to_industry_offset
        )
        live_preservation = preserve_live_from_support(
            output,
            live_support,
            combined_knot,
            live_label,
            source_mask=mask,
            source_to_output_matrix=industry_to_output_matrix,
            source_to_output_offset=industry_to_output_offset,
            stage_name="thesis_fit_block_160_by_80_by_80",
        )

    summary = {
        "restored_pith_slices": restored_pith_slices,
        "live_preservation": live_preservation,
        "live_voxels": int(np.count_nonzero(output == live_label)),
        "dead_voxels": int(np.count_nonzero(output == dead_label)),
        "pith_voxels": int(np.count_nonzero(output == PITH_LABEL)),
    }
    return output, summary


def validate_resampled_mask(
    source_labels: Sequence[int],
    output_mask: np.ndarray,
    knot_id: int,
) -> Tuple[List[int], List[str]]:
    output_labels = integer_labels(output_mask, "Final local mask")
    allowed = {
        BACKGROUND_LABEL,
        CLEAR_WOOD_LABEL,
        PITH_LABEL,
        knot_id,
        knot_id + DEAD_LABEL_OFFSET,
    }
    unexpected = sorted(set(output_labels) - allowed)
    if unexpected:
        raise ValueError(f"Final mask contains unexpected labels: {unexpected}")

    warnings = []
    if knot_id not in output_labels:
        message = f"Live label {knot_id} disappeared from the final block."
        if REQUIRE_LIVE_KNOT_AFTER_RESAMPLING:
            raise ValueError(message)
        warnings.append(message)

    if PITH_LABEL not in output_labels:
        message = f"Pith label {PITH_LABEL} disappeared from the final block."
        if REQUIRE_PITH_AFTER_RESAMPLING:
            raise ValueError(message)
        warnings.append(message)

    dead_label = knot_id + DEAD_LABEL_OFFSET
    if dead_label in source_labels and dead_label not in output_labels:
        warnings.append(
            f"Dead label {dead_label} disappeared at the coarse fixed grid."
        )

    return output_labels, warnings


# -----------------------------------------------------------------------------
# Output paths, headers, verification, and metadata
# -----------------------------------------------------------------------------


def output_paths(sample: KnotCropFiles) -> Dict[str, str]:
    tree_id = f"{sample.tree_number:02d}"
    disk_id = f"{sample.disk_number:02d}"
    knot_id = f"{sample.knot_id:02d}"
    prefix = (
        f"thesis_fit_Tree{tree_id}_Disk{disk_id}_Knot{knot_id}"
    )
    directory = os.path.join(
        OUTPUT_ROOT,
        f"Tree{tree_id}_thesis_fit",
        f"Disk{disk_id}_thesis_fit",
        f"Knot{knot_id}_thesis_fit",
    )

    return {
        "directory": directory,
        "dry_nhdr": os.path.join(directory, f"Dry_{prefix}.nhdr"),
        "dry_raw": os.path.join(directory, f"Dry_{prefix}.raw.gz"),
        "wet_nhdr": os.path.join(directory, f"Wet_{prefix}.nhdr"),
        "wet_raw": os.path.join(directory, f"Wet_{prefix}.raw.gz"),
        "mask_nhdr": os.path.join(directory, f"Mask_{prefix}.nhdr"),
        "mask_raw": os.path.join(directory, f"Mask_{prefix}.raw.gz"),
        "transform_json": os.path.join(
            directory,
            f"Transform_{prefix}.json",
        ),
    }


def any_output_exists(paths: Dict[str, str]) -> bool:
    return any(
        os.path.exists(paths[key])
        for key in (
            "dry_nhdr",
            "dry_raw",
            "wet_nhdr",
            "wet_raw",
            "mask_nhdr",
            "mask_raw",
            "transform_json",
        )
    )


def make_output_header(
    source_header: dict,
    transform: LocalBlockTransform,
    content: str,
) -> dict:
    header = copy.deepcopy(source_header)
    for field in HEADER_FIELDS_TO_REBUILD:
        header.pop(field, None)

    header["space directions"] = np.array(
        transform.target_directions,
        dtype=float,
        copy=True,
    )
    header["space origin"] = np.array(
        transform.target_origin,
        dtype=float,
        copy=True,
    )
    header["encoding"] = "gzip"
    header["content"] = content
    return header


def verify_gzip_file(path: str) -> None:
    with open(path, "rb") as input_file:
        signature = input_file.read(2)
    if signature != b"\x1f\x8b":
        raise ValueError(f"Detached file is not gzip compressed: {path}")


def write_detached_nhdr(
    nhdr_path: str,
    raw_path: str,
    data: np.ndarray,
    header: dict,
    transform: LocalBlockTransform,
    name: str,
) -> None:
    nrrd.write(
        nhdr_path,
        data,
        header=header,
        detached_header=raw_path,
        relative_data_path=True,
        compression_level=COMPRESSION_LEVEL,
        index_order=INDEX_ORDER,
    )

    if not os.path.isfile(nhdr_path) or not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Output pair was not created: {nhdr_path}")
    verify_gzip_file(raw_path)

    if not VERIFY_WRITTEN_FILES:
        return

    written_data, written_header = read_nhdr(nhdr_path)
    if written_data.shape != TARGET_BLOCK_SHAPE:
        raise ValueError(
            f"Written {name} shape is {written_data.shape}, expected "
            f"{TARGET_BLOCK_SHAPE}."
        )
    if written_data.dtype != data.dtype:
        raise TypeError(
            f"Written {name} type is {written_data.dtype}, expected "
            f"{data.dtype}."
        )
    if not np.array_equal(written_data, data):
        raise ValueError(f"Written {name} values differ from memory.")

    directions, _, origin = geometry_from_header(written_header, name)
    if not np.allclose(
        directions,
        transform.target_directions,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"Written {name} directions are incorrect.")
    if not np.allclose(
        origin,
        transform.target_origin,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(f"Written {name} origin is incorrect.")

    expected_raw_name = os.path.basename(raw_path)
    actual_raw_name = os.path.basename(
        str(written_header.get("data file", ""))
    )
    if actual_raw_name != expected_raw_name:
        raise ValueError(
            f"Written NHDR refers to {actual_raw_name}, expected "
            f"{expected_raw_name}."
        )


def array_list(values: np.ndarray) -> list:
    return np.asarray(values, dtype=float).tolist()


def transform_metadata(
    sample: KnotCropFiles,
    industry_grid: IndustryGrid,
    transform: LocalBlockTransform,
    source_labels: Sequence[int],
    output_labels: Sequence[int],
    industry_mask_summary: dict,
    local_mask_summary: dict,
) -> dict:
    scale_r, scale_t, scale_l = transform.scale_factors
    effective_r, effective_t, effective_l = transform.effective_spacing_mm

    return {
        "schema_version": 4,
        "replication_target": {
            "method": "study_shape_with_thesis_fit_policy",
            "version": "thesis_fit_v1",
            "stage": "step_2_knot_area_analysis",
            "implementation": (
                "padding_when_native_extent_fits_and_axis_specific_"
                "downsampling_only_when_oversized"
            ),
            "published_parameters": {
                "initial_ct_spacing_mm": list(INDUSTRY_SPACING_MM),
                "fixed_block_shape_r_t_z": list(TARGET_BLOCK_SHAPE),
                "radial_subblock_width": RADIAL_SUBBLOCK_WIDTH,
                "coarse_radial_step": COARSE_RADIAL_STEP,
                "reported_evaluations_per_knot": (
                    REPORTED_EVALUATIONS_PER_KNOT
                ),
            },
            "explicit_implementation_assumptions": [
                (
                    "This is a deliberate hybrid rather than an exact "
                    "published-study scaling replication. It uses the "
                    "study's 160 by 80 by 80 shape with the earlier thesis "
                    "replication fit policy."
                ),
                (
                    "An axis keeps scale factor 1.0 when its native extent "
                    "fits. Empty output space is padding. An axis is "
                    "downsampled only when its extent is too large."
                ),
                (
                    "The study does not publish the pith output index. "
                    f"This implementation uses radial index "
                    f"{RADIAL_PITH_OUTPUT_INDEX}."
                ),
                (
                    "The study does not publish its interpolation and "
                    "padding rules. This implementation uses linear image "
                    "sampling and occupancy-based label sampling."
                ),
                (
                    "If ordinary inverse sampling removes every live voxel, "
                    "the fallback forward maps genuine source live voxels "
                    "and "
                    "restores the best non-pith output candidate."
                ),
            ],
        },
        "label_preservation_policy": {
            "minimum_live_voxels": (
                MINIMUM_LIVE_VOXELS_AFTER_RESAMPLING
            ),
            "geometric_forward_restoration_enabled": (
                ENABLE_GEOMETRIC_FORWARD_RESTORATION
            ),
            "forward_neighbour_radius_voxels": (
                FORWARD_RESTORATION_NEIGHBOUR_RADIUS
            ),
            "pith_is_never_overwritten": True,
        },
        "tree": sample.tree_number,
        "disk": sample.disk_number,
        "knot_id": sample.knot_id,
        "source_files": {
            "dry_image": sample.dry_image,
            "wet_image": sample.wet_image,
            "shared_mask": sample.shared_mask,
        },
        "coordinate_order": ["radial", "tangential", "longitudinal"],
        "target_shape_voxels": list(TARGET_BLOCK_SHAPE),
        "radial_pith_output_index": RADIAL_PITH_OUTPUT_INDEX,
        "tangential_axis_output_index": TANGENTIAL_AXIS_OUTPUT_INDEX,
        "source_shape_voxels": list(industry_grid.source_shape),
        "source_spacing_mm": list(industry_grid.source_spacing),
        "industry_shape_voxels": list(industry_grid.output_shape),
        "industry_spacing_mm": list(industry_grid.target_spacing),
        "rotation_angle_radians": transform.angle_radians,
        "rotation_angle_degrees": math.degrees(transform.angle_radians),
        "radial_unit_vector_in_industry_index_plane": array_list(
            transform.radial_unit_index
        ),
        "tangential_unit_vector_in_industry_index_plane": array_list(
            transform.tangential_unit_index
        ),
        "pith_reference_industry_index": array_list(
            transform.pith_reference_industry_index
        ),
        "raw_local_knot_bounds_in_industry_voxels": {
            "radial_minimum": transform.raw_local_knot_bounds[0],
            "radial_maximum": transform.raw_local_knot_bounds[1],
            "tangential_minimum": transform.raw_local_knot_bounds[2],
            "tangential_maximum": transform.raw_local_knot_bounds[3],
            "longitudinal_minimum": transform.raw_local_knot_bounds[4],
            "longitudinal_maximum": transform.raw_local_knot_bounds[5],
        },
        "local_knot_bounds_in_industry_voxels": {
            "radial_minimum": transform.local_knot_bounds[0],
            "radial_maximum": transform.local_knot_bounds[1],
            "tangential_minimum": transform.local_knot_bounds[2],
            "tangential_maximum": transform.local_knot_bounds[3],
            "longitudinal_minimum": transform.local_knot_bounds[4],
            "longitudinal_maximum": transform.local_knot_bounds[5],
        },
        "required_native_shape_voxels": list(
            transform.required_native_shape
        ),
        "scale_factors": {
            "radial": scale_r,
            "tangential": scale_t,
            "longitudinal": scale_l,
        },
        "scale_factor_definition": (
            "industry-grid index displacement per one output voxel"
        ),
        "scale_modes": {
            "radial": scale_mode(scale_r),
            "tangential": scale_mode(scale_t),
            "longitudinal": scale_mode(scale_l),
        },
        "thesis_fit_policy": {
            "minimum_scale_factor": 1.0,
            "small_knot_action": "pad_without_enlarging",
            "oversized_axis_action": "downsample_only_that_axis",
            "radial_inner_margin_mm": RADIAL_INNER_MARGIN_MM,
            "radial_outer_margin_mm": RADIAL_OUTER_MARGIN_MM,
            "tangential_margin_mm": TANGENTIAL_MARGIN_MM,
            "longitudinal_margin_slices": LONGITUDINAL_MARGIN_SLICES,
        },
        "canonical_spacing_mm": list(transform.canonical_spacing_mm),
        "effective_output_spacing_mm": list(
            transform.effective_spacing_mm
        ),
        "padding_before_voxels": list(transform.padding_before),
        "padding_after_voxels": list(transform.padding_after),
        "output_to_industry_index_matrix": array_list(
            transform.output_to_industry_matrix
        ),
        "output_to_industry_index_offset": array_list(
            transform.output_to_industry_offset
        ),
        "output_to_original_source_index_matrix": array_list(
            transform.output_to_source_matrix
        ),
        "output_to_original_source_index_offset": array_list(
            transform.output_to_source_offset
        ),
        "output_space_directions": array_list(transform.target_directions),
        "output_space_origin": array_list(transform.target_origin),
        "source_labels": [int(value) for value in source_labels],
        "output_labels": [int(value) for value in output_labels],
        "industry_mask_summary": industry_mask_summary,
        "local_mask_summary": local_mask_summary,
        "prediction_conversion": {
            "dkb_mm_from_pith": (
                "radial_index_minus_radial_pith_output_index multiplied by "
                "effective_output_spacing_mm radial"
            ),
            "diameter_mm": (
                "diameter_in_output_tangential_voxels multiplied by "
                "effective_output_spacing_mm tangential"
            ),
        },
    }


def write_transform_json(path: str, metadata: dict) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    if VERIFY_WRITTEN_FILES:
        with open(path, "r", encoding="utf-8") as input_file:
            verified = json.load(input_file)
        if verified.get("target_shape_voxels") != list(TARGET_BLOCK_SHAPE):
            raise ValueError("Written transform JSON failed verification.")


def vector_text(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(v):.8g}" for v in values) + ")"


def scale_mode(scale: float, tolerance: float = 1e-8) -> str:
    """Describe an output-to-industry scale factor."""
    if scale > 1.0 + tolerance:
        return "downsampled"
    if scale < 1.0 - tolerance:
        return "upsampled"
    return "unchanged"


# -----------------------------------------------------------------------------
# Per knot processing
# -----------------------------------------------------------------------------


def process_one_crop(sample: KnotCropFiles) -> dict:
    paths = output_paths(sample)
    if any_output_exists(paths) and not OVERWRITE_EXISTING_OUTPUT:
        raise FileExistsError(
            "At least one output already exists and overwrite is disabled: "
            + paths["directory"]
        )

    dry_source, dry_header = read_nhdr(sample.dry_image)
    wet_source, wet_header = read_nhdr(sample.wet_image)
    mask_source, mask_header = read_nhdr(sample.shared_mask)

    validate_input_set(
        sample,
        dry_source,
        dry_header,
        wet_source,
        wet_header,
        mask_source,
        mask_header,
    )
    source_labels = validate_source_mask(mask_source, sample.knot_id)

    industry_grid = calculate_industry_grid(dry_source.shape, dry_header)
    dry_industry = resample_image_to_industry(dry_source, industry_grid)
    wet_industry = resample_image_to_industry(wet_source, industry_grid)
    try:
        mask_industry, industry_mask_summary = resample_mask_to_industry(
            mask_source,
            industry_grid,
            sample.knot_id,
        )
    except Exception as error:
        raise RuntimeError(
            "Mask stage 1 failed during conversion to the "
            "1 by 1 by 10 mm industry grid. "
            f"Original error: {error}"
        ) from error

    transform = build_local_transform(
        mask_industry,
        industry_grid,
        sample.knot_id,
    )
    coordinates = source_coordinates_for_block(transform)

    dry_output = resample_local_image(
        dry_industry,
        coordinates,
        background_fill_value(dry_industry, mask_industry),
    )
    wet_output = resample_local_image(
        wet_industry,
        coordinates,
        background_fill_value(wet_industry, mask_industry),
    )
    try:
        mask_output, local_mask_summary = resample_mask_to_local_block(
            mask_industry,
            coordinates,
            transform,
            sample.knot_id,
        )
    except Exception as error:
        raise RuntimeError(
            "Mask stage 2 failed during construction of the "
            "160 by 80 by 80 thesis-fit hybrid block. "
            f"Original error: {error}"
        ) from error
    output_labels, warnings = validate_resampled_mask(
        source_labels,
        mask_output,
        sample.knot_id,
    )

    if industry_mask_summary["live_preservation"]["applied"]:
        preservation = industry_mask_summary["live_preservation"]
        warnings.append(
            "Live tissue required preservation during the 1 by 1 by 10 mm "
            f"conversion using {preservation['method']}."
        )
    if local_mask_summary["live_preservation"]["applied"]:
        preservation = local_mask_summary["live_preservation"]
        warnings.append(
            "Live tissue required preservation during final local block "
            f"construction using {preservation['method']}."
        )

    identity = (
        f"tree {sample.tree_number:02d}, disk {sample.disk_number:02d}, "
        f"knot {sample.knot_id:02d}"
    )
    scale_text = vector_text(transform.scale_factors)
    dry_output_header = make_output_header(
        dry_header,
        transform,
        f"Study-shape thesis-fit radial tangential longitudinal dry block, "
        f"{identity}, "
        f"scale factors {scale_text}",
    )
    wet_output_header = make_output_header(
        wet_header,
        transform,
        f"Study-shape thesis-fit radial tangential longitudinal wet block, "
        f"{identity}, "
        f"scale factors {scale_text}",
    )
    mask_output_header = make_output_header(
        mask_header,
        transform,
        f"Study-shape thesis-fit radial tangential longitudinal shared mask, "
        f"{identity}, "
        f"scale factors {scale_text}",
    )

    metadata = transform_metadata(
        sample,
        industry_grid,
        transform,
        source_labels,
        output_labels,
        industry_mask_summary,
        local_mask_summary,
    )

    os.makedirs(paths["directory"], exist_ok=True)
    write_detached_nhdr(
        paths["dry_nhdr"],
        paths["dry_raw"],
        dry_output,
        dry_output_header,
        transform,
        "dry image",
    )
    write_detached_nhdr(
        paths["wet_nhdr"],
        paths["wet_raw"],
        wet_output,
        wet_output_header,
        transform,
        "wet image",
    )
    write_detached_nhdr(
        paths["mask_nhdr"],
        paths["mask_raw"],
        mask_output,
        mask_output_header,
        transform,
        "shared mask",
    )
    write_transform_json(paths["transform_json"], metadata)

    scale_r, scale_t, scale_l = transform.scale_factors
    spacing_r, spacing_t, spacing_l = transform.effective_spacing_mm
    return {
        "status": "created",
        "tree": f"{sample.tree_number:02d}",
        "disk": sample.disk_number,
        "knot_id": sample.knot_id,
        "source_shape": str(industry_grid.source_shape),
        "industry_shape": str(industry_grid.output_shape),
        "required_local_shape": str(transform.required_native_shape),
        "output_shape": str(TARGET_BLOCK_SHAPE),
        "rotation_angle_degrees": math.degrees(transform.angle_radians),
        "scale_radial": scale_r,
        "scale_tangential": scale_t,
        "scale_longitudinal": scale_l,
        "scale_mode_radial": scale_mode(scale_r),
        "scale_mode_tangential": scale_mode(scale_t),
        "scale_mode_longitudinal": scale_mode(scale_l),
        "effective_radial_spacing_mm": spacing_r,
        "effective_tangential_spacing_mm": spacing_t,
        "effective_longitudinal_spacing_mm": spacing_l,
        "padding_before": str(transform.padding_before),
        "padding_after": str(transform.padding_after),
        "source_labels": str(source_labels),
        "output_labels": str(output_labels),
        "output_live_voxels": local_mask_summary["live_voxels"],
        "output_dead_voxels": local_mask_summary["dead_voxels"],
        "output_pith_voxels": local_mask_summary["pith_voxels"],
        "industry_live_preservation": industry_mask_summary[
            "live_preservation"
        ]["applied"],
        "industry_live_preservation_method": industry_mask_summary[
            "live_preservation"
        ]["method"],
        "industry_live_restored_voxels": industry_mask_summary[
            "live_preservation"
        ]["restored_voxels"],
        "local_live_preservation": local_mask_summary[
            "live_preservation"
        ]["applied"],
        "local_live_preservation_method": local_mask_summary[
            "live_preservation"
        ]["method"],
        "local_live_restored_voxels": local_mask_summary[
            "live_preservation"
        ]["restored_voxels"],
        "warnings": " ".join(warnings),
        "dry_image": paths["dry_nhdr"],
        "wet_image": paths["wet_nhdr"],
        "shared_mask": paths["mask_nhdr"],
        "transform_json": paths["transform_json"],
        "message": "",
    }


MANIFEST_FIELDS = [
    "status",
    "tree",
    "disk",
    "knot_id",
    "source_shape",
    "industry_shape",
    "required_local_shape",
    "output_shape",
    "rotation_angle_degrees",
    "scale_radial",
    "scale_tangential",
    "scale_longitudinal",
    "scale_mode_radial",
    "scale_mode_tangential",
    "scale_mode_longitudinal",
    "effective_radial_spacing_mm",
    "effective_tangential_spacing_mm",
    "effective_longitudinal_spacing_mm",
    "padding_before",
    "padding_after",
    "source_labels",
    "output_labels",
    "output_live_voxels",
    "output_dead_voxels",
    "output_pith_voxels",
    "industry_live_preservation",
    "industry_live_preservation_method",
    "industry_live_restored_voxels",
    "local_live_preservation",
    "local_live_preservation_method",
    "local_live_restored_voxels",
    "warnings",
    "dry_image",
    "wet_image",
    "shared_mask",
    "transform_json",
    "message",
]


def status_row(sample: KnotCropFiles, status: str, message: str) -> dict:
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        {
            "status": status,
            "tree": f"{sample.tree_number:02d}",
            "disk": sample.disk_number,
            "knot_id": sample.knot_id,
            "output_shape": str(TARGET_BLOCK_SHAPE),
            "message": message,
        }
    )
    return row


def write_manifest(
    rows: List[dict],
    first_tree_number: int,
    last_tree_number: int,
) -> str:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    path = os.path.join(
        OUTPUT_ROOT,
        f"Trees{first_tree_number:02d}_to_{last_tree_number:02d}_"
        "manifest_thesis_fit.csv",
    )
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    validate_settings()
    if REFUSE_EXISTING_OUTPUT_ROOT and os.path.exists(OUTPUT_ROOT):
        raise FileExistsError(
            "Hybrid output root already exists and will not be overwritten: "
            f"{OUTPUT_ROOT}. Change OUTPUT_ROOT to a new folder such as "
            "Individual_Knot_Crops_160x80x80_thesis_fit_v2."
        )

    tree_numbers = list(range(FIRST_TREE_NUMBER, LAST_TREE_NUMBER + 1))
    samples: List[KnotCropFiles] = []
    samples_per_tree: Dict[int, int] = {}
    missing_trees = []

    for tree_number in tree_numbers:
        try:
            tree_samples = find_knot_crops(SOURCE_ROOT, tree_number)
        except FileNotFoundError as error:
            print(f"WARNING | {error}")
            missing_trees.append(tree_number)
            samples_per_tree[tree_number] = 0
            continue

        samples_per_tree[tree_number] = len(tree_samples)
        samples.extend(tree_samples)

    if not samples:
        print("No complete knot crop sets were found.")
        return

    if REFUSE_EXISTING_OUTPUT_ROOT:
        os.makedirs(OUTPUT_ROOT, exist_ok=False)
    else:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(
        f"Found {len(samples)} complete knot crops for Trees "
        f"{FIRST_TREE_NUMBER:02d} through {LAST_TREE_NUMBER:02d}."
    )
    print(f"Source root: {SOURCE_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("Method: study shape with thesis fitting policy")
    print("Final array order: radial, tangential, longitudinal")
    print(f"Final shape: {TARGET_BLOCK_SHAPE}")
    print(f"Initial industry spacing: {INDUSTRY_SPACING_MM} mm")
    print(
        "An axis keeps scale 1.0 when it fits and receives padding. Only "
        "an oversized axis is downsampled. Every scale factor is saved."
    )
    print(
        "Published downstream settings: radial subblock width "
        f"{RADIAL_SUBBLOCK_WIDTH}, coarse step "
        f"{COARSE_RADIAL_STEP}."
    )
    print(
        "Geometric forward label restoration is enabled with neighbour "
        f"radius {FORWARD_RESTORATION_NEIGHBOUR_RADIUS}."
    )

    rows: List[dict] = []
    for index, sample in enumerate(samples, start=1):
        identity = (
            f"Tree {sample.tree_number:02d} | "
            f"Disk {sample.disk_number:02d} | "
            f"Knot {sample.knot_id:02d}"
        )
        print(f"\n{index} of {len(samples)} | {identity}")

        try:
            row = process_one_crop(sample)
            rows.append(row)
            print(
                f"CREATED | local {row['required_local_shape']} to "
                f"{row['output_shape']} | scales "
                f"{row['scale_radial']:.5g}, "
                f"{row['scale_tangential']:.5g}, "
                f"{row['scale_longitudinal']:.5g}"
            )
            if row["warnings"]:
                print(f"WARNING | {row['warnings']}")

        except FileExistsError as error:
            rows.append(status_row(sample, "skipped", str(error)))
            print(f"SKIPPED | {error}")

        except Exception as error:
            rows.append(status_row(sample, "failed", str(error)))
            print(f"FAILED | {error}")

    manifest_path = write_manifest(
        rows,
        FIRST_TREE_NUMBER,
        LAST_TREE_NUMBER,
    )

    created_count = sum(row["status"] == "created" for row in rows)
    skipped_count = sum(row["status"] == "skipped" for row in rows)
    failed_count = sum(row["status"] == "failed" for row in rows)
    radial_upsampled = sum(
        row.get("scale_mode_radial") == "upsampled" for row in rows
    )
    radial_downsampled = sum(
        row.get("scale_mode_radial") == "downsampled" for row in rows
    )
    tangential_upsampled = sum(
        row.get("scale_mode_tangential") == "upsampled" for row in rows
    )
    tangential_downsampled = sum(
        row.get("scale_mode_tangential") == "downsampled" for row in rows
    )
    longitudinal_upsampled = sum(
        row.get("scale_mode_longitudinal") == "upsampled" for row in rows
    )
    longitudinal_downsampled = sum(
        row.get("scale_mode_longitudinal") == "downsampled" for row in rows
    )
    industry_forward_restored = sum(
        row.get("industry_live_preservation_method")
        in (
            "geometric_forward_mapping",
            "inverse_then_geometric_forward_mapping",
        )
        for row in rows
    )
    local_forward_restored = sum(
        row.get("local_live_preservation_method")
        in (
            "geometric_forward_mapping",
            "inverse_then_geometric_forward_mapping",
        )
        for row in rows
    )

    print("\nFinished")
    print(f"Knot crops found: {len(samples)}")
    print(f"Created: {created_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(
        "Radial scale modes: "
        f"{radial_upsampled} upsampled, "
        f"{radial_downsampled} downsampled"
    )
    print(
        "Tangential scale modes: "
        f"{tangential_upsampled} upsampled, "
        f"{tangential_downsampled} downsampled"
    )
    print(
        "Longitudinal scale modes: "
        f"{longitudinal_upsampled} upsampled, "
        f"{longitudinal_downsampled} downsampled"
    )
    print(
        "Geometric live restoration: "
        f"{industry_forward_restored} industry blocks, "
        f"{local_forward_restored} final hybrid blocks"
    )
    if missing_trees:
        print(
            "Missing tree folders: "
            + ", ".join(f"{value:02d}" for value in missing_trees)
        )
    print(f"Manifest: {manifest_path}")


def cli(argv=None):
    import argparse
    global SOURCE_ROOT, OUTPUT_ROOT, FIRST_TREE_NUMBER, LAST_TREE_NUMBER
    parser = argparse.ArgumentParser(description='Resample, align and fit knot crops into 160 x 80 x 80 blocks.')
    parser.add_argument('--source-root', default=SOURCE_ROOT)
    parser.add_argument('--output-root', default=OUTPUT_ROOT)
    parser.add_argument('--first-tree', type=int, default=FIRST_TREE_NUMBER)
    parser.add_argument('--last-tree', type=int, default=LAST_TREE_NUMBER)
    args = parser.parse_args(argv)
    SOURCE_ROOT, OUTPUT_ROOT = args.source_root, args.output_root
    FIRST_TREE_NUMBER, LAST_TREE_NUMBER = args.first_tree, args.last_tree
    if not 1 <= FIRST_TREE_NUMBER <= LAST_TREE_NUMBER <= 24:
        parser.error('Tree range must be within 1 through 24')
    paths.require_input_directory(SOURCE_ROOT)
    paths.require_new_output(OUTPUT_ROOT)
    main()


if __name__ == "__main__":
    cli()
