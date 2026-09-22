"""Crop individual knots for a selectable inclusive range of pine trees.

The script processes one tree and one disk at a time, so running a large tree
range does not keep all volumes in memory. Each tree receives its own manifest,
and one combined manifest is written for the complete run.
"""

import paths as paths
import copy
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import nrrd
import numpy as np


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

FIRST_TREE_NUMBER = 1
LAST_TREE_NUMBER = 24
NUMBER_OF_DISKS = 16

# Input folders are described in docs/data_format.md.
SOURCE_TREE_ROOT = str(paths.SOURCE_ROOT)
PITH_ROOT = str(paths.PITH_ROOT)

# A completely new output tree is created here.
OUTPUT_ROOT = str(paths.CROP_ROOT)

# Axis 2 is assumed to be the through-slice axis. Change this only if the
# slices in your arrays are stored along another NumPy axis.
SLICE_AXIS = 2
SLICE_MARGIN = 5

# The code starts with this much margin in the other two axes. It reduces the
# margin down to zero when doing so can prevent another knot entering the crop.
MAX_IN_PLANE_MARGIN = 10

PITH_CLASS = 150
BACKGROUND_LABEL = 0
LIVE_LABEL_MIN = 1
LIVE_LABEL_MAX = 50
DEAD_LABEL_OFFSET = 50
DEAD_LABEL_MIN = LIVE_LABEL_MIN + DEAD_LABEL_OFFSET
DEAD_LABEL_MAX = LIVE_LABEL_MAX + DEAD_LABEL_OFFSET
CLEAR_WOOD_LABEL = 255

INDEX_ORDER = "F"
COMPRESSION_LEVEL = 6

# Existing crop files are protected by default.
OVERWRITE_EXISTING_OUTPUT = False

# A common mask is valid only when dry and wet images use the same grid.
REQUIRE_MATCHING_IMAGE_GEOMETRY = True

# This guarantees that the pith runs through every slice in the saved crop.
REQUIRE_PITH_IN_EVERY_CROP_SLICE = True

# Reading each saved file back is slower, but confirms that no values were
# changed or transposed during writing.
VERIFY_WRITTEN_FILES = True


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
    "sample units",
    "axis mins",
    "axis maxs",
    "axismins",
    "axismaxs",
}


@dataclass(frozen=True)
class DiskInputs:
    disk_number: int
    dry_image: str
    wet_image: str
    dry_mask: str
    pith_mask: str


@dataclass(frozen=True)
class CropBounds:
    starts: Tuple[int, int, int]
    stops: Tuple[int, int, int]
    in_plane_margin: int
    other_knot_voxels: int
    other_knot_ids: Tuple[int, ...]

    @property
    def slices(self) -> Tuple[slice, slice, slice]:
        return tuple(
            slice(start, stop)
            for start, stop in zip(self.starts, self.stops)
        )

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(
            stop - start
            for start, stop in zip(self.starts, self.stops)
        )


def tree_directory(tree_number: int) -> str:
    """Return the full image and instance mask folder for one tree."""

    return os.path.join(
        SOURCE_TREE_ROOT,
        f"PINE Tree {tree_number}_Finished",
    )


def pith_directory(tree_number: int) -> str:
    """Return the pith mask folder for one tree."""

    return os.path.join(PITH_ROOT, f"Tree{tree_number}")


def build_disk_inputs(
    tree_number: int,
    disk_number: int,
    tree_dir: str,
    pith_dir: str,
) -> DiskInputs:
    tree_id = f"{tree_number:02d}"
    disk_id = str(disk_number)

    return DiskInputs(
        disk_number=disk_number,
        dry_image=os.path.join(
            tree_dir,
            f"Drydisk{tree_id}.{disk_id}.nhdr",
        ),
        wet_image=os.path.join(
            tree_dir,
            f"Green Disk_{tree_id}.{disk_id}.nhdr",
        ),
        dry_mask=os.path.join(
            tree_dir,
            f"SEG_Drydisk{tree_id}.{disk_id}.nhdr",
        ),
        pith_mask=os.path.join(
            pith_dir,
            f"pith_SEG_Green Disk_{tree_id}.{disk_id}.nhdr",
        ),
    )


def read_nhdr(path: str):
    if not path.lower().endswith(".nhdr"):
        raise ValueError(f"Only NHDR input is allowed: {path}")

    return nrrd.read(path, index_order=INDEX_ORDER)


def validate_input_files(inputs: DiskInputs) -> None:
    nhdr_paths = (
        inputs.dry_image,
        inputs.wet_image,
        inputs.dry_mask,
        inputs.pith_mask,
    )
    required_files = []

    for nhdr_path in nhdr_paths:
        required_files.append(nhdr_path)
        required_files.append(nhdr_path[:-5] + ".raw.gz")

    missing = [
        path for path in required_files if not os.path.isfile(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing input files: " + ", ".join(missing)
        )


def validate_common_shape(
    dry_image: np.ndarray,
    wet_image: np.ndarray,
    dry_mask: np.ndarray,
    pith_mask: np.ndarray,
    disk_number: int,
) -> None:
    shapes = {
        "dry image": dry_image.shape,
        "wet image": wet_image.shape,
        "dry mask": dry_mask.shape,
        "pith mask": pith_mask.shape,
    }

    if len(set(shapes.values())) != 1:
        shape_text = ", ".join(
            f"{name}={shape}" for name, shape in shapes.items()
        )
        raise ValueError(
            f"Disk {disk_number} has mismatched shapes: {shape_text}"
        )

    if dry_image.ndim != 3:
        raise ValueError(
            f"Disk {disk_number} is not three dimensional: {dry_image.shape}"
        )

    if SLICE_AXIS not in (0, 1, 2):
        raise ValueError("SLICE_AXIS must be 0, 1, or 2.")


def header_values_match(first, second) -> bool:
    try:
        first_array = np.asarray(first)
        second_array = np.asarray(second)

        if first_array.shape != second_array.shape:
            return False

        if np.issubdtype(first_array.dtype, np.number) and np.issubdtype(
            second_array.dtype,
            np.number,
        ):
            return bool(
                np.allclose(
                    first_array.astype(float),
                    second_array.astype(float),
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
            )

        return bool(np.array_equal(first_array, second_array))
    except (TypeError, ValueError):
        return first == second


def validate_registered_image_geometry(
    dry_header: dict,
    wet_header: dict,
    disk_number: int,
) -> None:
    geometry_fields = (
        "space",
        "space directions",
        "space origin",
        "spacings",
    )
    differences = []

    for field in geometry_fields:
        dry_has_field = field in dry_header
        wet_has_field = field in wet_header

        if dry_has_field != wet_has_field:
            differences.append(f"{field} is present in only one header")
            continue

        if dry_has_field and not header_values_match(
            dry_header[field],
            wet_header[field],
        ):
            differences.append(f"{field} differs")

    if not differences:
        return

    message = (
        f"Disk {disk_number} dry and wet image geometry differs: "
        + "; ".join(differences)
    )

    if REQUIRE_MATCHING_IMAGE_GEOMETRY:
        raise ValueError(message)

    print(f"WARNING: {message}")


def validate_instance_mask(mask_data: np.ndarray, path: str) -> List[int]:
    if not np.issubdtype(mask_data.dtype, np.integer):
        if not np.issubdtype(mask_data.dtype, np.floating):
            raise TypeError(f"Mask has a nonnumeric type: {mask_data.dtype}")

        if not np.all(np.isfinite(mask_data)):
            raise ValueError(f"Mask contains nonfinite values: {path}")

        if not np.all(mask_data == np.round(mask_data)):
            raise ValueError(f"Mask contains noninteger values: {path}")

    labels = sorted(np.unique(mask_data).astype(int).tolist())
    valid_labels = {
        BACKGROUND_LABEL,
        CLEAR_WOOD_LABEL,
        *range(LIVE_LABEL_MIN, LIVE_LABEL_MAX + 1),
        *range(DEAD_LABEL_MIN, DEAD_LABEL_MAX + 1),
    }
    unexpected = [label for label in labels if label not in valid_labels]

    if unexpected:
        raise ValueError(
            f"Unexpected mask labels {unexpected} in {path}"
        )

    return labels


def find_knot_ids(labels: Sequence[int]):
    live_ids = {
        label
        for label in labels
        if LIVE_LABEL_MIN <= label <= LIVE_LABEL_MAX
    }
    dead_ids = {
        label - DEAD_LABEL_OFFSET
        for label in labels
        if DEAD_LABEL_MIN <= label <= DEAD_LABEL_MAX
    }

    dead_without_live = sorted(dead_ids - live_ids)
    knot_ids = sorted(live_ids)

    return knot_ids, dead_ids, dead_without_live


def make_pith_binary(pith_data: np.ndarray) -> np.ndarray:
    return (
        (pith_data != BACKGROUND_LABEL)
        & (pith_data != CLEAR_WOOD_LABEL)
    )


def make_target_knot_mask(
    instance_mask: np.ndarray,
    knot_id: int,
) -> np.ndarray:
    live_label = knot_id
    dead_label = knot_id + DEAD_LABEL_OFFSET

    return (
        (instance_mask == live_label)
        | (instance_mask == dead_label)
    )


def make_other_knots_mask(
    instance_mask: np.ndarray,
    knot_id: int,
) -> np.ndarray:
    all_knot_voxels = (
        (
            (instance_mask >= LIVE_LABEL_MIN)
            & (instance_mask <= LIVE_LABEL_MAX)
        )
        | (
            (instance_mask >= DEAD_LABEL_MIN)
            & (instance_mask <= DEAD_LABEL_MAX)
        )
    )

    return all_knot_voxels & ~make_target_knot_mask(
        instance_mask,
        knot_id,
    )


def label_to_knot_id(label: int) -> int:
    if DEAD_LABEL_MIN <= label <= DEAD_LABEL_MAX:
        return label - DEAD_LABEL_OFFSET
    return label


def get_other_knot_ids(
    instance_mask_crop: np.ndarray,
    knot_id: int,
) -> Tuple[int, ...]:
    labels = np.unique(instance_mask_crop).astype(int).tolist()
    other_ids = {
        label_to_knot_id(label)
        for label in labels
        if (
            LIVE_LABEL_MIN <= label <= LIVE_LABEL_MAX
            or DEAD_LABEL_MIN <= label <= DEAD_LABEL_MAX
        )
        and label_to_knot_id(label) != knot_id
    }

    return tuple(sorted(other_ids))


def bounds_with_margin(
    mandatory_min: np.ndarray,
    mandatory_max: np.ndarray,
    volume_shape: Tuple[int, int, int],
    in_plane_margin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    starts = mandatory_min.copy()
    ends_inclusive = mandatory_max.copy()

    for axis in range(3):
        if axis == SLICE_AXIS:
            continue

        starts[axis] = max(0, starts[axis] - in_plane_margin)
        ends_inclusive[axis] = min(
            volume_shape[axis] - 1,
            ends_inclusive[axis] + in_plane_margin,
        )

    stops = ends_inclusive + 1
    return starts, stops


def calculate_crop_bounds(
    instance_mask: np.ndarray,
    pith_binary: np.ndarray,
    knot_id: int,
) -> CropBounds:
    target_mask = make_target_knot_mask(instance_mask, knot_id)
    target_coordinates = np.argwhere(target_mask)

    if target_coordinates.size == 0:
        raise ValueError(f"Knot {knot_id} has no voxels.")

    target_min = target_coordinates.min(axis=0)
    target_max = target_coordinates.max(axis=0)

    slice_start = max(
        0,
        int(target_min[SLICE_AXIS]) - SLICE_MARGIN,
    )
    slice_end_inclusive = min(
        instance_mask.shape[SLICE_AXIS] - 1,
        int(target_max[SLICE_AXIS]) + SLICE_MARGIN,
    )

    pith_coordinates = np.argwhere(pith_binary)
    pith_coordinates = pith_coordinates[
        (pith_coordinates[:, SLICE_AXIS] >= slice_start)
        & (pith_coordinates[:, SLICE_AXIS] <= slice_end_inclusive)
    ]

    if pith_coordinates.size == 0:
        raise ValueError(
            f"Knot {knot_id} has no pith voxels within its slice range."
        )

    mandatory_min = target_min.copy()
    mandatory_max = target_max.copy()

    for axis in range(3):
        if axis == SLICE_AXIS:
            mandatory_min[axis] = slice_start
            mandatory_max[axis] = slice_end_inclusive
        else:
            mandatory_min[axis] = min(
                int(target_min[axis]),
                int(pith_coordinates[:, axis].min()),
            )
            mandatory_max[axis] = max(
                int(target_max[axis]),
                int(pith_coordinates[:, axis].max()),
            )

    selected_starts = None
    selected_stops = None
    selected_margin = 0
    selected_other_count = 0

    for margin in range(MAX_IN_PLANE_MARGIN, -1, -1):
        starts, stops = bounds_with_margin(
            mandatory_min=mandatory_min,
            mandatory_max=mandatory_max,
            volume_shape=instance_mask.shape,
            in_plane_margin=margin,
        )
        crop_slices = tuple(
            slice(start, stop)
            for start, stop in zip(starts, stops)
        )
        other_count = int(
            np.count_nonzero(
                make_other_knots_mask(
                    instance_mask[crop_slices],
                    knot_id,
                )
            )
        )

        selected_starts = starts
        selected_stops = stops
        selected_margin = margin
        selected_other_count = other_count

        if other_count == 0:
            break

    selected_slices = tuple(
        slice(start, stop)
        for start, stop in zip(selected_starts, selected_stops)
    )
    other_ids = get_other_knot_ids(
        instance_mask[selected_slices],
        knot_id,
    )

    return CropBounds(
        starts=tuple(int(value) for value in selected_starts),
        stops=tuple(int(value) for value in selected_stops),
        in_plane_margin=selected_margin,
        other_knot_voxels=selected_other_count,
        other_knot_ids=other_ids,
    )


def validate_pith_crop(
    pith_crop: np.ndarray,
    knot_id: int,
) -> None:
    if not np.any(pith_crop):
        raise ValueError(f"Knot {knot_id} crop does not contain pith.")

    in_plane_axes = tuple(
        axis for axis in range(3) if axis != SLICE_AXIS
    )
    pith_present_by_slice = np.any(
        pith_crop,
        axis=in_plane_axes,
    )

    missing_relative_slices = np.flatnonzero(~pith_present_by_slice)

    if (
        REQUIRE_PITH_IN_EVERY_CROP_SLICE
        and missing_relative_slices.size > 0
    ):
        raise ValueError(
            f"Knot {knot_id} crop has no pith in relative slices "
            f"{missing_relative_slices.tolist()}."
        )


def create_shared_crop_mask(
    instance_mask_crop: np.ndarray,
    pith_crop: np.ndarray,
    knot_id: int,
) -> np.ndarray:
    """Keep one target knot and the pith in a common dry and wet mask."""

    output_mask = np.full(
        instance_mask_crop.shape,
        CLEAR_WOOD_LABEL,
        dtype=np.uint8,
    )

    output_mask[instance_mask_crop == BACKGROUND_LABEL] = BACKGROUND_LABEL

    live_label = knot_id
    dead_label = knot_id + DEAD_LABEL_OFFSET

    output_mask[instance_mask_crop == live_label] = live_label
    output_mask[instance_mask_crop == dead_label] = dead_label

    # Pith has priority where the original knot and pith masks overlap.
    output_mask[pith_crop] = PITH_CLASS

    return output_mask


def direction_vector_for_axis(directions, axis: int):
    try:
        vector = directions[axis]
    except (TypeError, IndexError):
        return None

    if vector is None:
        return None

    try:
        vector_array = np.asarray(vector, dtype=float)
    except (TypeError, ValueError):
        return None

    if vector_array.ndim != 1 or not np.all(np.isfinite(vector_array)):
        return None

    return vector_array


def update_crop_origin(
    output_header: dict,
    crop_starts: Tuple[int, int, int],
) -> None:
    origin = output_header.get("space origin")
    directions = output_header.get("space directions")

    if origin is None or directions is None:
        return

    try:
        new_origin = np.asarray(origin, dtype=float).copy()
    except (TypeError, ValueError):
        return

    for axis, start in enumerate(crop_starts):
        direction = direction_vector_for_axis(directions, axis)
        if direction is None or direction.shape != new_origin.shape:
            continue
        new_origin += int(start) * direction

    output_header["space origin"] = new_origin


def make_crop_header(
    source_header: dict,
    crop_starts: Tuple[int, int, int],
    content: str,
) -> dict:
    output_header = copy.deepcopy(source_header)

    for field in HEADER_FIELDS_TO_REBUILD:
        output_header.pop(field, None)

    output_header["encoding"] = "gzip"
    output_header["content"] = content
    update_crop_origin(output_header, crop_starts)

    return output_header


def build_output_paths(
    tree_number: int,
    disk_number: int,
    knot_id: int,
) -> Dict[str, str]:
    tree_id = f"{tree_number:02d}"
    disk_id = f"{disk_number:02d}"
    knot_text = f"{knot_id:02d}"

    knot_dir = os.path.join(
        OUTPUT_ROOT,
        f"Tree{tree_id}",
        f"Disk{disk_id}",
        f"Knot{knot_text}",
    )
    prefix = f"Tree{tree_id}_Disk{disk_id}_Knot{knot_text}"

    dry_base = os.path.join(knot_dir, f"Dry_{prefix}")
    wet_base = os.path.join(knot_dir, f"Wet_{prefix}")
    mask_base = os.path.join(knot_dir, f"Mask_{prefix}")

    return {
        "directory": knot_dir,
        "dry_nhdr": dry_base + ".nhdr",
        "dry_raw": dry_base + ".raw.gz",
        "wet_nhdr": wet_base + ".nhdr",
        "wet_raw": wet_base + ".raw.gz",
        "mask_nhdr": mask_base + ".nhdr",
        "mask_raw": mask_base + ".raw.gz",
    }


def outputs_already_exist(output_paths: Dict[str, str]) -> bool:
    file_keys = (
        "dry_nhdr",
        "dry_raw",
        "wet_nhdr",
        "wet_raw",
        "mask_nhdr",
        "mask_raw",
    )
    return any(os.path.exists(output_paths[key]) for key in file_keys)


def verify_gzip_file(raw_path: str) -> None:
    with open(raw_path, "rb") as raw_file:
        signature = raw_file.read(2)

    if signature != b"\x1f\x8b":
        raise ValueError(f"File is not gzip compressed: {raw_path}")


def write_detached_nhdr(
    nhdr_path: str,
    raw_path: str,
    data: np.ndarray,
    header: dict,
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
        raise FileNotFoundError(
            f"Expected output pair was not created: {nhdr_path}"
        )

    verify_gzip_file(raw_path)

    if not VERIFY_WRITTEN_FILES:
        return

    written_data, written_header = nrrd.read(
        nhdr_path,
        index_order=INDEX_ORDER,
    )

    if written_data.dtype != data.dtype:
        raise TypeError(
            f"Written type {written_data.dtype} does not match {data.dtype}: "
            f"{nhdr_path}"
        )

    if not np.array_equal(written_data, data):
        raise ValueError(f"Written values do not match: {nhdr_path}")

    expected_raw_name = os.path.basename(raw_path)
    written_raw_name = os.path.basename(
        str(written_header.get("data file", ""))
    )

    if expected_raw_name != written_raw_name:
        raise ValueError(
            f"NHDR references '{written_raw_name}' instead of "
            f"'{expected_raw_name}'."
        )


def write_knot_outputs(
    dry_crop: np.ndarray,
    wet_crop: np.ndarray,
    shared_mask_crop: np.ndarray,
    dry_header: dict,
    wet_header: dict,
    crop_bounds: CropBounds,
    tree_number: int,
    disk_number: int,
    knot_id: int,
    condition: str,
) -> Dict[str, str]:
    output_paths = build_output_paths(
        tree_number=tree_number,
        disk_number=disk_number,
        knot_id=knot_id,
    )

    if outputs_already_exist(output_paths) and not OVERWRITE_EXISTING_OUTPUT:
        raise FileExistsError(
            "At least one output already exists and overwrite is disabled: "
            + output_paths["directory"]
        )

    os.makedirs(output_paths["directory"], exist_ok=True)

    identity = (
        f"tree {tree_number:02d}, disk {disk_number}, knot {knot_id}, "
        f"{condition}"
    )

    dry_crop_header = make_crop_header(
        source_header=dry_header,
        crop_starts=crop_bounds.starts,
        content=f"Dry individual knot crop, {identity}",
    )
    wet_crop_header = make_crop_header(
        source_header=wet_header,
        crop_starts=crop_bounds.starts,
        content=f"Wet individual knot crop, {identity}",
    )

    # The image grids have already been checked as identical. The dry image
    # geometry is therefore used for the one mask shared by both images.
    shared_mask_header = make_crop_header(
        source_header=dry_header,
        crop_starts=crop_bounds.starts,
        content=f"Shared knot and pith mask, {identity}",
    )

    write_detached_nhdr(
        nhdr_path=output_paths["dry_nhdr"],
        raw_path=output_paths["dry_raw"],
        data=dry_crop,
        header=dry_crop_header,
    )
    write_detached_nhdr(
        nhdr_path=output_paths["wet_nhdr"],
        raw_path=output_paths["wet_raw"],
        data=wet_crop,
        header=wet_crop_header,
    )
    write_detached_nhdr(
        nhdr_path=output_paths["mask_nhdr"],
        raw_path=output_paths["mask_raw"],
        data=shared_mask_crop,
        header=shared_mask_header,
    )

    return output_paths


def process_one_knot(
    dry_image: np.ndarray,
    wet_image: np.ndarray,
    instance_mask: np.ndarray,
    pith_binary: np.ndarray,
    dry_header: dict,
    wet_header: dict,
    tree_number: int,
    disk_number: int,
    knot_id: int,
    dead_ids: set,
) -> dict:
    crop_bounds = calculate_crop_bounds(
        instance_mask=instance_mask,
        pith_binary=pith_binary,
        knot_id=knot_id,
    )
    crop_slices = crop_bounds.slices

    dry_crop = np.array(dry_image[crop_slices], copy=True)
    wet_crop = np.array(wet_image[crop_slices], copy=True)
    instance_mask_crop = instance_mask[crop_slices]
    pith_crop = pith_binary[crop_slices]

    validate_pith_crop(pith_crop, knot_id)

    shared_mask_crop = create_shared_crop_mask(
        instance_mask_crop=instance_mask_crop,
        pith_crop=pith_crop,
        knot_id=knot_id,
    )

    condition = (
        "live_and_dead" if knot_id in dead_ids else "fully_live"
    )

    output_paths = write_knot_outputs(
        dry_crop=dry_crop,
        wet_crop=wet_crop,
        shared_mask_crop=shared_mask_crop,
        dry_header=dry_header,
        wet_header=wet_header,
        crop_bounds=crop_bounds,
        tree_number=tree_number,
        disk_number=disk_number,
        knot_id=knot_id,
        condition=condition,
    )

    pith_voxels = int(np.count_nonzero(pith_crop))
    target_voxels = int(
        np.count_nonzero(
            make_target_knot_mask(instance_mask_crop, knot_id)
        )
    )

    return {
        "status": "created",
        "tree": f"{tree_number:02d}",
        "disk": disk_number,
        "knot_id": knot_id,
        "condition": condition,
        "live_label": knot_id,
        "dead_label": (
            knot_id + DEAD_LABEL_OFFSET
            if knot_id in dead_ids
            else ""
        ),
        "crop_start": str(crop_bounds.starts),
        "crop_stop_exclusive": str(crop_bounds.stops),
        "crop_shape": str(crop_bounds.shape),
        "slice_axis": SLICE_AXIS,
        "slice_margin": SLICE_MARGIN,
        "in_plane_margin_used": crop_bounds.in_plane_margin,
        "target_knot_voxels": target_voxels,
        "pith_voxels": pith_voxels,
        "other_knot_ids_in_image_crop": ",".join(
            str(value) for value in crop_bounds.other_knot_ids
        ),
        "other_knot_voxels_in_image_crop": crop_bounds.other_knot_voxels,
        "dry_image": output_paths["dry_nhdr"],
        "wet_image": output_paths["wet_nhdr"],
        "shared_mask": output_paths["mask_nhdr"],
        "message": "",
    }


def failed_manifest_row(
    tree_number: int,
    disk_number: int,
    knot_id,
    message: str,
) -> dict:
    return {
        "status": "failed",
        "tree": f"{tree_number:02d}",
        "disk": disk_number,
        "knot_id": knot_id,
        "condition": "",
        "live_label": "",
        "dead_label": "",
        "crop_start": "",
        "crop_stop_exclusive": "",
        "crop_shape": "",
        "slice_axis": SLICE_AXIS,
        "slice_margin": SLICE_MARGIN,
        "in_plane_margin_used": "",
        "target_knot_voxels": "",
        "pith_voxels": "",
        "other_knot_ids_in_image_crop": "",
        "other_knot_voxels_in_image_crop": "",
        "dry_image": "",
        "wet_image": "",
        "shared_mask": "",
        "message": message,
    }


def process_one_disk(
    inputs: DiskInputs,
    tree_number: int,
) -> List[dict]:
    validate_input_files(inputs)

    dry_image, dry_header = read_nhdr(inputs.dry_image)
    wet_image, wet_header = read_nhdr(inputs.wet_image)
    instance_mask, _ = read_nhdr(inputs.dry_mask)
    pith_data, _ = read_nhdr(inputs.pith_mask)

    validate_common_shape(
        dry_image=dry_image,
        wet_image=wet_image,
        dry_mask=instance_mask,
        pith_mask=pith_data,
        disk_number=inputs.disk_number,
    )
    validate_registered_image_geometry(
        dry_header=dry_header,
        wet_header=wet_header,
        disk_number=inputs.disk_number,
    )

    labels = validate_instance_mask(instance_mask, inputs.dry_mask)
    knot_ids, dead_ids, dead_without_live = find_knot_ids(labels)
    pith_binary = make_pith_binary(pith_data)

    print(f"Disk {inputs.disk_number} labels: {labels}")
    print(f"Disk {inputs.disk_number} knot IDs: {knot_ids}")

    rows: List[dict] = []

    for knot_id in dead_without_live:
        message = (
            f"Dead label {knot_id + DEAD_LABEL_OFFSET} exists without "
            f"live label {knot_id}. The knot was not cropped."
        )
        print(f"WARNING: {message}")
        rows.append(
            failed_manifest_row(
                tree_number=tree_number,
                disk_number=inputs.disk_number,
                knot_id=knot_id,
                message=message,
            )
        )

    for knot_id in knot_ids:
        try:
            row = process_one_knot(
                dry_image=dry_image,
                wet_image=wet_image,
                instance_mask=instance_mask,
                pith_binary=pith_binary,
                dry_header=dry_header,
                wet_header=wet_header,
                tree_number=tree_number,
                disk_number=inputs.disk_number,
                knot_id=knot_id,
                dead_ids=dead_ids,
            )
            rows.append(row)

            overlap_text = ""
            if row["other_knot_voxels_in_image_crop"]:
                overlap_text = (
                    " | unavoidable nearby knots "
                    + row["other_knot_ids_in_image_crop"]
                )

            print(
                f"CREATED | Disk {inputs.disk_number:2d} | "
                f"Knot {knot_id:2d} | {row['condition']} | "
                f"shape {row['crop_shape']}{overlap_text}"
            )

        except FileExistsError as error:
            row = failed_manifest_row(
                tree_number=tree_number,
                disk_number=inputs.disk_number,
                knot_id=knot_id,
                message=str(error),
            )
            row["status"] = "skipped"
            rows.append(row)
            print(
                f"SKIPPED | Disk {inputs.disk_number:2d} | "
                f"Knot {knot_id:2d} | {error}"
            )

        except Exception as error:
            rows.append(
                failed_manifest_row(
                    tree_number=tree_number,
                    disk_number=inputs.disk_number,
                    knot_id=knot_id,
                    message=str(error),
                )
            )
            print(
                f"FAILED  | Disk {inputs.disk_number:2d} | "
                f"Knot {knot_id:2d} | {error}"
            )

    return rows


def manifest_fieldnames() -> List[str]:
    return [
        "status",
        "tree",
        "disk",
        "knot_id",
        "condition",
        "live_label",
        "dead_label",
        "crop_start",
        "crop_stop_exclusive",
        "crop_shape",
        "slice_axis",
        "slice_margin",
        "in_plane_margin_used",
        "target_knot_voxels",
        "pith_voxels",
        "other_knot_ids_in_image_crop",
        "other_knot_voxels_in_image_crop",
        "dry_image",
        "wet_image",
        "shared_mask",
        "message",
    ]


def write_manifest(rows: List[dict], tree_number: int) -> str:
    tree_output_dir = os.path.join(
        OUTPUT_ROOT,
        f"Tree{tree_number:02d}",
    )
    os.makedirs(tree_output_dir, exist_ok=True)

    manifest_path = os.path.join(
        tree_output_dir,
        f"Tree{tree_number:02d}_crop_manifest.csv",
    )

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=manifest_fieldnames(),
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def summarize_rows(rows: List[dict]) -> dict:
    """Count the result categories in a list of manifest rows."""

    return {
        "created": sum(row["status"] == "created" for row in rows),
        "skipped": sum(row["status"] == "skipped" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "unavoidable": sum(
            int(row["other_knot_voxels_in_image_crop"] or 0) > 0
            for row in rows
        ),
    }


def write_combined_manifest(rows: List[dict]) -> str:
    """Write one manifest containing results from every requested tree."""

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    manifest_path = os.path.join(
        OUTPUT_ROOT,
        (
            f"Trees{FIRST_TREE_NUMBER:02d}_to_"
            f"{LAST_TREE_NUMBER:02d}_crop_manifest.csv"
        ),
    )

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=manifest_fieldnames(),
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def process_one_tree(tree_number: int) -> Tuple[List[dict], str]:
    """Crop every available knot for one tree and write its manifest."""

    tree_dir = tree_directory(tree_number)
    pith_dir = pith_directory(tree_number)

    if not os.path.isdir(tree_dir):
        raise FileNotFoundError(f"Tree folder does not exist: {tree_dir}")

    if not os.path.isdir(pith_dir):
        raise FileNotFoundError(f"Pith folder does not exist: {pith_dir}")

    print("\n" + "=" * 88)
    print(f"CROPPING INDIVIDUAL KNOTS FOR PINE TREE {tree_number:02d}")
    print(f"Tree folder: {tree_dir}")
    print(f"Pith folder: {pith_dir}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Slice axis: {SLICE_AXIS}")
    print(f"Slices retained before and after each knot: {SLICE_MARGIN}")
    print("The source files will not be modified.")
    print("=" * 88)

    tree_rows: List[dict] = []

    for disk_number in range(1, NUMBER_OF_DISKS + 1):
        inputs = build_disk_inputs(
            tree_number=tree_number,
            disk_number=disk_number,
            tree_dir=tree_dir,
            pith_dir=pith_dir,
        )

        try:
            disk_rows = process_one_disk(
                inputs=inputs,
                tree_number=tree_number,
            )
            tree_rows.extend(disk_rows)
        except Exception as error:
            print(f"FAILED  | Disk {disk_number:2d} | {error}")
            tree_rows.append(
                failed_manifest_row(
                    tree_number=tree_number,
                    disk_number=disk_number,
                    knot_id="",
                    message=str(error),
                )
            )

    manifest_path = write_manifest(tree_rows, tree_number)
    summary = summarize_rows(tree_rows)

    print("\n" + "=" * 88)
    print(f"TREE {tree_number:02d} SUMMARY")
    print("=" * 88)
    print(f"Created knot crops: {summary['created']}")
    print(f"Skipped knot crops: {summary['skipped']}")
    print(f"Failed knot crops:  {summary['failed']}")
    print(
        "Crops with unavoidable nearby knot voxels: "
        f"{summary['unavoidable']}"
    )
    print(f"Tree manifest: {manifest_path}")

    return tree_rows, manifest_path


def main() -> None:
    if FIRST_TREE_NUMBER > LAST_TREE_NUMBER:
        raise ValueError(
            "FIRST_TREE_NUMBER cannot be greater than LAST_TREE_NUMBER."
        )

    if FIRST_TREE_NUMBER < 1 or LAST_TREE_NUMBER > 24:
        raise ValueError("The pine tree range must remain between 1 and 24.")

    requested_trees = list(
        range(FIRST_TREE_NUMBER, LAST_TREE_NUMBER + 1)
    )

    print("=" * 88)
    print(
        "MULTI TREE INDIVIDUAL KNOT CROPPING | "
        f"TREES {FIRST_TREE_NUMBER:02d} TO {LAST_TREE_NUMBER:02d}"
    )
    print(f"Trees to process: {requested_trees}")
    print(f"Source root: {SOURCE_TREE_ROOT}")
    print(f"Pith root: {PITH_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("Trees and disks are processed sequentially to control memory use.")
    print("A failure in one tree or disk does not stop the remaining range.")
    print("=" * 88)

    all_rows: List[dict] = []
    completed_trees = 0
    tree_level_failures = 0

    for tree_number in requested_trees:
        try:
            tree_rows, _ = process_one_tree(tree_number)
            completed_trees += 1
        except Exception as error:
            tree_level_failures += 1
            message = str(error)
            print("\n" + "!" * 88)
            print(f"TREE {tree_number:02d} FAILED | {message}")
            print("Continuing with the next tree.")
            print("!" * 88)

            tree_rows = [
                failed_manifest_row(
                    tree_number=tree_number,
                    disk_number="",
                    knot_id="",
                    message=message,
                )
            ]
            write_manifest(tree_rows, tree_number)

        all_rows.extend(tree_rows)

    combined_manifest = write_combined_manifest(all_rows)
    summary = summarize_rows(all_rows)

    print("\n" + "=" * 88)
    print("COMPLETE RANGE SUMMARY")
    print("=" * 88)
    print(f"Trees requested: {len(requested_trees)}")
    print(f"Trees processed: {completed_trees}")
    print(f"Tree level failures: {tree_level_failures}")
    print(f"Created knot crops: {summary['created']}")
    print(f"Skipped knot crops: {summary['skipped']}")
    print(f"Failed rows: {summary['failed']}")
    print(
        "Crops with unavoidable nearby knot voxels: "
        f"{summary['unavoidable']}"
    )
    print(f"Combined manifest: {combined_manifest}")


def cli(argv=None):
    import argparse
    global SOURCE_TREE_ROOT, PITH_ROOT, OUTPUT_ROOT, FIRST_TREE_NUMBER, LAST_TREE_NUMBER, NUMBER_OF_DISKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default=SOURCE_TREE_ROOT)
    parser.add_argument('--pith-root', default=PITH_ROOT)
    parser.add_argument('--output-root', default=OUTPUT_ROOT)
    parser.add_argument('--first-tree', type=int, default=FIRST_TREE_NUMBER)
    parser.add_argument('--last-tree', type=int, default=LAST_TREE_NUMBER)
    parser.add_argument('--disks', type=int, default=NUMBER_OF_DISKS)
    args = parser.parse_args(argv)
    SOURCE_TREE_ROOT, PITH_ROOT, OUTPUT_ROOT = args.source_root, args.pith_root, args.output_root
    FIRST_TREE_NUMBER, LAST_TREE_NUMBER, NUMBER_OF_DISKS = args.first_tree, args.last_tree, args.disks
    if NUMBER_OF_DISKS < 1:
        parser.error('--disks must be positive')
    paths.require_input_directory(SOURCE_TREE_ROOT)
    paths.require_input_directory(PITH_ROOT)
    paths.require_new_output(OUTPUT_ROOT)
    main()


if __name__ == "__main__":
    cli()
