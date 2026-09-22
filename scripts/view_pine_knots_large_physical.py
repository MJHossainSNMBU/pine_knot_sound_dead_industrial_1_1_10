"""Large physical-scale VTK viewer for original, resampled and study-fit knots.

Install: python -m pip install numpy pynrrd vtk
Run this file directly. Set paths with --original-root, --previous-root and --block-root.

Rows: original, previous 1 x 1 x 10 mm, thesis-fit 160 x 80 x 80.
Columns: wet image, wet mask, dry image, dry mask.

Left drag rotates all four panels in the hovered row. Middle drag pans that row.
Wheel/right drag zooms. In equal-scale mode, zoom changes scale across all rows.
M toggles equal mm scale across rows / individually fitted rows.
1/2/3 enlarges one row; 0 restores all three rows. R refits visible rows.
F toggles fullscreen. Arrows/Space change knot. L/D/P toggle mask labels.
B toggles tight display crop versus full stored volume. Q/Escape exits.

Default framing uses the target knot AND pith bounds with a 3 mm margin.
This is an in-memory display crop only. No source file is modified.
All actors use NHDR spacing, directions and origin, including effective spacing
for downsampled axes. Only cameras magnify the data; geometry is not stretched.
A separate camera is shared by the four panels in each row. Default orthographic
projection gives the same displayed length per millimetre across visible rows.
In independent fit mode magnification can differ by row; each has a scale bar.
No display interpolation can restore detail lost during earlier resampling.
"""

from __future__ import annotations

import paths as paths

import csv
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import nrrd
import numpy as np
import vtk
from vtkmodules.util import numpy_support


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

FIRST_TREE_NUMBER = 1
LAST_TREE_NUMBER = 24

ORIGINAL_ROOT = str(paths.CROP_ROOT)

PREVIOUS_RESAMPLED_ROOT = str(paths.PREVIOUS_ROOT)



THESIS_FIT_ROOT = str(paths.BLOCK_ROOT)

STEM_AXIS = 2
ORIGINAL_EXPECTED_SPACING_MM = (0.5, 0.5, 0.5)
RESAMPLED_IN_PLANE_SPACING_MM = 1.0
RESAMPLED_STEM_SPACING_MM = 10.0
THESIS_FIT_SHAPE = (160, 80, 80)

SPACING_TOLERANCE_MM = 1e-5
DIRECTION_TOLERANCE = 1e-6
CENTER_TOLERANCE_MM = 1e-4
INDEX_ORDER = "F"

WINDOW_SIZE = (1800, 1000)
START_FULL_SCREEN = True
IMAGE_THRESHOLD = 800

NUMBER_OF_ROWS = 3
NUMBER_OF_COLUMNS = 4


# -----------------------------------------------------------------------------
# Mask display settings
# -----------------------------------------------------------------------------

BACKGROUND_LABEL = 0
LIVE_LABEL_MIN = 1
LIVE_LABEL_MAX = 50
DEAD_LABEL_OFFSET = 50
PITH_LABEL = 150
CLEAR_WOOD_LABEL = 255

DISPLAY_LIVE_VALUE = 1
DISPLAY_DEAD_VALUE = 2
DISPLAY_PITH_VALUE = 3

LIVE_COLOR = (0.0, 1.0, 0.0)
DEAD_COLOR = (1.0, 0.0, 0.0)
PITH_COLOR = (1.0, 1.0, 0.0)

LIVE_OPACITY = 0.95
DEAD_OPACITY = 0.35
PITH_OPACITY = 0.95

MASK_RENDER_ORDER = (
    DISPLAY_DEAD_VALUE,
    DISPLAY_LIVE_VALUE,
    DISPLAY_PITH_VALUE,
)


@dataclass(frozen=True)
class ResolutionFiles:
    dry_image: str
    wet_image: str
    shared_mask: str


@dataclass(frozen=True)
class ComparisonSample:
    tree_number: int
    disk_number: int
    knot_id: int
    original: ResolutionFiles
    previous_resampling: ResolutionFiles
    thesis_fit: ResolutionFiles


@dataclass(frozen=True)
class PhysicalGeometry:
    directions: np.ndarray
    spacing: np.ndarray
    unit_directions: np.ndarray
    origin: np.ndarray


@dataclass(frozen=True)
class LoadedResolution:
    dry_data: np.ndarray
    wet_data: np.ndarray
    mask_data: np.ndarray
    geometry: PhysicalGeometry


@dataclass(frozen=True)
class ResolutionOutcome:
    loaded: Optional[LoadedResolution]
    reason: str


def numbered_subdirectories(parent: str, prefix: str):
    """Return folders such as Disk02 and Knot03 in numerical order."""

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    matches = []

    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        match = pattern.fullmatch(name)
        if match and os.path.isdir(path):
            matches.append((int(match.group(1)), path))

    return sorted(matches, key=lambda item: item[0])


def files_for_knot(
    root: str,
    tree_number: int,
    disk_number: int,
    knot_id: int,
) -> ResolutionFiles:
    knot_directory = os.path.join(
        root,
        f"Tree{tree_number:02d}",
        f"Disk{disk_number:02d}",
        f"Knot{knot_id:02d}",
    )
    identity = (
        f"Tree{tree_number:02d}_Disk{disk_number:02d}_Knot{knot_id:02d}"
    )

    return ResolutionFiles(
        dry_image=os.path.join(knot_directory, f"Dry_{identity}.nhdr"),
        wet_image=os.path.join(knot_directory, f"Wet_{identity}.nhdr"),
        shared_mask=os.path.join(knot_directory, f"Mask_{identity}.nhdr"),
    )


def thesis_fit_files_for_knot(
    tree_number: int,
    disk_number: int,
    knot_id: int,
) -> ResolutionFiles:
    """Build paths written by the 160 by 80 by 80 thesis-fit script."""

    knot_directory = os.path.join(
        THESIS_FIT_ROOT,
        f"Tree{tree_number:02d}_thesis_fit",
        f"Disk{disk_number:02d}_thesis_fit",
        f"Knot{knot_id:02d}_thesis_fit",
    )
    identity = (
        f"thesis_fit_Tree{tree_number:02d}_Disk{disk_number:02d}_"
        f"Knot{knot_id:02d}"
    )

    return ResolutionFiles(
        dry_image=os.path.join(knot_directory, f"Dry_{identity}.nhdr"),
        wet_image=os.path.join(knot_directory, f"Wet_{identity}.nhdr"),
        shared_mask=os.path.join(knot_directory, f"Mask_{identity}.nhdr"),
    )


def detached_pair_items(
    files: ResolutionFiles,
) -> Tuple[Tuple[str, str], ...]:
    return (
        ("dry image NHDR", files.dry_image),
        ("dry image raw.gz", files.dry_image[:-5] + ".raw.gz"),
        ("wet image NHDR", files.wet_image),
        ("wet image raw.gz", files.wet_image[:-5] + ".raw.gz"),
        ("mask NHDR", files.shared_mask),
        ("mask raw.gz", files.shared_mask[:-5] + ".raw.gz"),
    )


def incomplete_set_reason(
    files: ResolutionFiles,
    manifest_record: Optional[dict],
) -> str:
    missing_names = [
        name
        for name, path in detached_pair_items(files)
        if not os.path.isfile(path)
    ]

    if not missing_names:
        return ""

    reason_parts = []
    if manifest_record:
        status = str(manifest_record.get("status", "")).strip()
        message = str(manifest_record.get("message", "")).strip()
        warnings = str(manifest_record.get("warnings", "")).strip()

        if status:
            reason_parts.append(f"Manifest status: {status.upper()}")
        if message:
            reason_parts.append(message)
        elif warnings:
            reason_parts.append(warnings)

    if not reason_parts:
        reason_parts.append(
            "No matching failure entry was found in the resampling manifest."
        )

    if len(missing_names) == len(detached_pair_items(files)):
        reason_parts.append("No output image or mask files were created.")
    else:
        reason_parts.append(
            "Missing files: " + ", ".join(missing_names) + "."
        )

    return "\n".join(reason_parts)


def load_manifest_records(
    root: str,
) -> Dict[Tuple[int, int, int], dict]:
    """Read every resampling manifest found below one output root."""

    records: Dict[Tuple[int, int, int], dict] = {}
    if not os.path.isdir(root):
        return records

    manifest_paths = []
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            lower_name = filename.lower()
            if lower_name.endswith(".csv") and "manifest" in lower_name:
                manifest_paths.append(os.path.join(directory, filename))

    manifest_paths.sort(key=lambda path: (os.path.getmtime(path), path))

    for manifest_path in manifest_paths:
        try:
            with open(
                manifest_path,
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as input_file:
                for row in csv.DictReader(input_file):
                    try:
                        key = (
                            int(str(row.get("tree", "")).strip()),
                            int(str(row.get("disk", "")).strip()),
                            int(str(row.get("knot_id", "")).strip()),
                        )
                    except (TypeError, ValueError):
                        continue

                    records[key] = dict(row)
        except (OSError, csv.Error) as error:
            print(f"WARNING | Could not read manifest {manifest_path}: {error}")

    return records


def find_samples() -> List[ComparisonSample]:
    """Use complete original knot crops as the master comparison list."""

    if FIRST_TREE_NUMBER > LAST_TREE_NUMBER:
        raise ValueError(
            "FIRST_TREE_NUMBER cannot be greater than LAST_TREE_NUMBER."
        )

    samples: List[ComparisonSample] = []

    for tree_number in range(FIRST_TREE_NUMBER, LAST_TREE_NUMBER + 1):
        original_tree_directory = os.path.join(
            ORIGINAL_ROOT,
            f"Tree{tree_number:02d}",
        )
        if not os.path.isdir(original_tree_directory):
            print(
                f"WARNING | Original Tree {tree_number:02d} folder is "
                f"missing: {original_tree_directory}"
            )
            continue

        for disk_number, disk_directory in numbered_subdirectories(
            original_tree_directory,
            "Disk",
        ):
            for knot_id, _ in numbered_subdirectories(
                disk_directory,
                "Knot",
            ):
                original = files_for_knot(
                    ORIGINAL_ROOT,
                    tree_number,
                    disk_number,
                    knot_id,
                )
                missing_original = [
                    path
                    for _, path in detached_pair_items(original)
                    if not os.path.isfile(path)
                ]

                if missing_original:
                    print(
                        f"Skipping Tree {tree_number:02d}, "
                        f"Disk {disk_number:02d}, Knot {knot_id:02d} "
                        "because its original detached set is incomplete."
                    )
                    continue

                samples.append(
                    ComparisonSample(
                        tree_number=tree_number,
                        disk_number=disk_number,
                        knot_id=knot_id,
                        original=original,
                        previous_resampling=files_for_knot(
                            PREVIOUS_RESAMPLED_ROOT,
                            tree_number,
                            disk_number,
                            knot_id,
                        ),
                        thesis_fit=thesis_fit_files_for_knot(
                            tree_number,
                            disk_number,
                            knot_id,
                        ),
                    )
                )

    return samples


def read_nhdr(path: str):
    if not path.lower().endswith(".nhdr"):
        raise ValueError(f"Only NHDR input is allowed: {path}")

    return nrrd.read(path, index_order=INDEX_ORDER)


def geometry_from_header(header: dict, name: str) -> PhysicalGeometry:
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
            f"{name} directions have shape {directions.shape}, not 3 by 3."
        )
    if origin.shape != (3,):
        raise ValueError(f"{name} origin does not contain three values.")
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(origin)):
        raise ValueError(f"{name} has nonfinite physical geometry values.")

    spacing = np.linalg.norm(directions, axis=1)
    if np.any(spacing <= 0):
        raise ValueError(f"{name} has a zero length direction vector.")

    return PhysicalGeometry(
        directions=directions,
        spacing=spacing,
        unit_directions=directions / spacing[:, None],
        origin=origin,
    )


def resampled_expected_spacing() -> np.ndarray:
    if STEM_AXIS not in (0, 1, 2):
        raise ValueError("STEM_AXIS must be 0, 1, or 2.")

    spacing = np.full(3, RESAMPLED_IN_PLANE_SPACING_MM, dtype=float)
    spacing[STEM_AXIS] = RESAMPLED_STEM_SPACING_MM
    return spacing


def load_resolution(
    files: ResolutionFiles,
    name: str,
    expected_spacing: Optional[Sequence[float]],
    expected_shape: Optional[Sequence[int]] = None,
    minimum_spacing: Optional[Sequence[float]] = None,
) -> LoadedResolution:
    dry_data, dry_header = read_nhdr(files.dry_image)
    wet_data, wet_header = read_nhdr(files.wet_image)
    mask_data, mask_header = read_nhdr(files.shared_mask)

    arrays = {
        "dry image": dry_data,
        "wet image": wet_data,
        "shared mask": mask_data,
    }
    if any(data.ndim != 3 for data in arrays.values()):
        raise ValueError(f"{name} does not contain three dimensional data.")
    if len({data.shape for data in arrays.values()}) != 1:
        shapes = ", ".join(
            f"{item_name}={data.shape}"
            for item_name, data in arrays.items()
        )
        raise ValueError(f"{name} shapes do not match. {shapes}")

    if expected_shape is not None:
        required_shape = tuple(int(value) for value in expected_shape)
        if dry_data.shape != required_shape:
            raise ValueError(
                f"{name} shape is {dry_data.shape}. Expected "
                f"{required_shape}."
            )

    dry_geometry = geometry_from_header(dry_header, f"{name} dry image")
    wet_geometry = geometry_from_header(wet_header, f"{name} wet image")
    mask_geometry = geometry_from_header(mask_header, f"{name} mask")
    expected = (
        None
        if expected_spacing is None
        else np.asarray(expected_spacing, dtype=float)
    )
    minimum = (
        None
        if minimum_spacing is None
        else np.asarray(minimum_spacing, dtype=float)
    )

    for item_name, geometry in (
        ("dry image", dry_geometry),
        ("wet image", wet_geometry),
        ("mask", mask_geometry),
    ):
        if expected is not None and not np.allclose(
            geometry.spacing,
            expected,
            rtol=0.0,
            atol=SPACING_TOLERANCE_MM,
        ):
            raise ValueError(
                f"{name} {item_name} spacing is "
                f"{geometry.spacing.tolist()} mm. Expected "
                f"{expected.tolist()} mm."
            )
        if minimum is not None and np.any(
            geometry.spacing < minimum - SPACING_TOLERANCE_MM
        ):
            raise ValueError(
                f"{name} {item_name} spacing is "
                f"{geometry.spacing.tolist()} mm. Every value must be at "
                f"least {minimum.tolist()} mm."
            )

    for item_name, geometry in (
        ("wet image", wet_geometry),
        ("mask", mask_geometry),
    ):
        if not np.allclose(
            dry_geometry.directions,
            geometry.directions,
            rtol=0.0,
            atol=DIRECTION_TOLERANCE,
        ):
            raise ValueError(
                f"{name} dry image and {item_name} directions do not match."
            )
        if not np.allclose(
            dry_geometry.origin,
            geometry.origin,
            rtol=0.0,
            atol=CENTER_TOLERANCE_MM,
        ):
            raise ValueError(
                f"{name} dry image and {item_name} origins do not match."
            )

    return LoadedResolution(
        dry_data=dry_data,
        wet_data=wet_data,
        mask_data=mask_data,
        geometry=dry_geometry,
    )


def load_resolution_outcome(
    files: ResolutionFiles,
    name: str,
    expected_spacing: Optional[Sequence[float]],
    manifest_record: Optional[dict],
    expected_shape: Optional[Sequence[int]] = None,
    minimum_spacing: Optional[Sequence[float]] = None,
) -> ResolutionOutcome:
    missing_reason = incomplete_set_reason(files, manifest_record)
    if missing_reason:
        return ResolutionOutcome(loaded=None, reason=missing_reason)

    try:
        loaded = load_resolution(
            files,
            name,
            expected_spacing,
            expected_shape=expected_shape,
            minimum_spacing=minimum_spacing,
        )
    except Exception as error:
        return ResolutionOutcome(
            loaded=None,
            reason=f"Files exist but could not be displayed.\n{error}",
        )

    return ResolutionOutcome(loaded=loaded, reason="")


def physical_center(
    shape: Sequence[int],
    geometry: PhysicalGeometry,
) -> np.ndarray:
    center = np.array(geometry.origin, dtype=float, copy=True)
    for axis, axis_size in enumerate(shape):
        center += 0.5 * (int(axis_size) - 1) * geometry.directions[axis]
    return center


def cross_resolution_error(
    original: LoadedResolution,
    comparison: LoadedResolution,
) -> str:
    if not np.allclose(
        original.geometry.unit_directions,
        comparison.geometry.unit_directions,
        rtol=0.0,
        atol=DIRECTION_TOLERANCE,
    ):
        return "Physical axis directions do not match the original crop."

    original_center = physical_center(
        original.dry_data.shape,
        original.geometry,
    )
    comparison_center = physical_center(
        comparison.dry_data.shape,
        comparison.geometry,
    )
    if not np.allclose(
        original_center,
        comparison_center,
        rtol=0.0,
        atol=CENTER_TOLERANCE_MM,
    ):
        difference = comparison_center - original_center
        return (
            "Physical centre does not match the original crop. "
            f"Difference in mm: {difference.tolist()}"
        )

    return ""





# Framing settings are in physical millimetres, not voxel counts.
DISPLAY_MARGIN_MM = 3.0
EQUAL_MM_SCALE = True
TIGHT_DISPLAY_CROP = True
FRAME_FILL = 0.86


def numpy_volume_to_vtk(data, geometry):
    """Use local millimetres plus an explicit actor transform.

    Keeping image axes local avoids relying on a CPU volume mapper to interpret
    VTK direction matrices. The actor transform restores the NHDR world frame.
    Singleton axes are duplicated for rendering, keeping their physical centre.
    """
    display = data
    local_origin = np.zeros(3)
    for axis in range(3):
        if display.shape[axis] == 1:
            display = np.repeat(display, 2, axis=axis)
            local_origin[axis] = -0.5 * geometry.spacing[axis]
    image = vtk.vtkImageData()
    image.SetDimensions(*display.shape)
    image.SetSpacing(*geometry.spacing)
    image.SetOrigin(*local_origin)
    values = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(display.ravel(order="F")), deep=True)
    image.GetPointData().SetScalars(values)
    pipeline = vtk.vtkPassThrough()
    pipeline.SetInputData(image)
    pipeline.Update()
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for row in range(3):
        for col in range(3):
            matrix.SetElement(row, col, float(geometry.unit_directions[col, row]))
        matrix.SetElement(row, 3, float(geometry.origin[row]))
    pipeline.world_matrix = matrix
    return pipeline


def crop_for_display(loaded, knot_id, tight=True):
    """Crop all three arrays equally, using tissue support rather than padding."""
    mask = loaded.mask_data
    support = (mask == knot_id) | (mask == knot_id + DEAD_LABEL_OFFSET) | (mask == PITH_LABEL)
    lo = np.zeros(3, dtype=int)
    hi = np.asarray(mask.shape, dtype=int)
    if tight and np.any(support):
        margin = np.ceil(DISPLAY_MARGIN_MM / loaded.geometry.spacing).astype(int)
        for axis in range(3):
            active = np.flatnonzero(np.any(support, axis=tuple(a for a in range(3) if a != axis)))
            lo[axis] = max(0, int(active[0]) - int(margin[axis]))
            hi[axis] = min(mask.shape[axis], int(active[-1]) + int(margin[axis]) + 1)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    g = loaded.geometry
    geometry = PhysicalGeometry(g.directions, g.spacing, g.unit_directions,
                                g.origin + lo @ g.directions)
    return LoadedResolution(loaded.dry_data[slices], loaded.wet_data[slices],
                            mask[slices], geometry)


def physical_corners(loaded):
    """Voxel edge corners in world mm, including the thickness of single slices."""
    import itertools
    edges = [(-0.5, n - 0.5) for n in loaded.mask_data.shape]
    indices = np.array(list(itertools.product(*edges)))
    return loaded.geometry.origin + indices @ loaded.geometry.directions


def create_image_volume(passthrough):
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(passthrough.GetOutputPort())
    threshold.ThresholdByUpper(IMAGE_THRESHOLD)
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0)

    mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    mapper.SetInputConnection(threshold.GetOutputPort())

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0, 0.0)
    opacity.AddPoint(1, 0.08)
    opacity.AddPoint(800, 0.10)
    opacity.AddPoint(2000, 0.15)

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    color.AddRGBPoint(1800, 1.0, 1.0, 1.0)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(opacity)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)
    volume.SetUserMatrix(passthrough.world_matrix)
    return volume


def create_class_volume(
    grouped_passthrough,
    class_value: int,
    color_rgb: Tuple[float, float, float],
    opacity_value: float,
):
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(grouped_passthrough.GetOutputPort())
    threshold.ThresholdBetween(class_value, class_value)
    threshold.ReplaceInOn()
    threshold.SetInValue(1)
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0)
    threshold.SetOutputScalarTypeToUnsignedChar()

    mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    mapper.SetInputConnection(threshold.GetOutputPort())

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0, 0.0)
    opacity.AddPoint(1, opacity_value)

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    color.AddRGBPoint(1, *color_rgb)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(opacity)
    volume_property.ShadeOff()
    volume_property.SetInterpolationTypeToNearest()

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)
    volume.SetUserMatrix(grouped_passthrough.world_matrix)
    return volume


def grouped_mask_pipeline(
    mask_data: np.ndarray,
    geometry: PhysicalGeometry,
    knot_id: int,
):
    if not LIVE_LABEL_MIN <= knot_id <= LIVE_LABEL_MAX:
        raise ValueError(f"Knot ID {knot_id} is outside the range 1 to 50.")

    if np.issubdtype(mask_data.dtype, np.floating):
        if not np.all(np.isfinite(mask_data)):
            raise ValueError("Mask contains nonfinite values.")
        if not np.all(mask_data == np.round(mask_data)):
            raise ValueError("Mask contains noninteger values.")

    grouped = np.zeros(mask_data.shape, dtype=np.uint8)
    grouped[mask_data == knot_id] = DISPLAY_LIVE_VALUE
    grouped[mask_data == knot_id + DEAD_LABEL_OFFSET] = DISPLAY_DEAD_VALUE
    grouped[mask_data == PITH_LABEL] = DISPLAY_PITH_VALUE
    return numpy_volume_to_vtk(grouped, geometry)


def create_mask_actor_set(grouped_passthrough):
    return {
        DISPLAY_LIVE_VALUE: create_class_volume(
            grouped_passthrough,
            DISPLAY_LIVE_VALUE,
            LIVE_COLOR,
            LIVE_OPACITY,
        ),
        DISPLAY_DEAD_VALUE: create_class_volume(
            grouped_passthrough,
            DISPLAY_DEAD_VALUE,
            DEAD_COLOR,
            DEAD_OPACITY,
        ),
        DISPLAY_PITH_VALUE: create_class_volume(
            grouped_passthrough,
            DISPLAY_PITH_VALUE,
            PITH_COLOR,
            PITH_OPACITY,
        ),
    }


def make_text_actor(text: str, font_size: int):
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()

    text_property = actor.GetTextProperty()
    text_property.SetFontFamilyToArial()
    text_property.SetFontSize(font_size)
    text_property.SetColor(0.0, 0.0, 0.0)
    text_property.SetVerticalJustificationToTop()
    return actor


def make_title_actor(text: str):
    actor = make_text_actor(text, font_size=14)
    actor.SetPosition(0.02, 0.98)
    return actor


def make_missing_actor(reason: str):
    wrapped_lines = []
    for paragraph in reason.splitlines():
        wrapped_lines.extend(textwrap.wrap(paragraph, width=38) or [""])

    actor = make_text_actor(
        "MISSING\n\n" + "\n".join(wrapped_lines),
        font_size=14,
    )
    actor.SetPosition(0.50, 0.62)

    text_property = actor.GetTextProperty()
    text_property.SetColor(0.65, 0.05, 0.05)
    text_property.SetBold(True)
    text_property.SetJustificationToCentered()
    text_property.SetVerticalJustificationToCentered()
    return actor




class ThreeResolutionKnotViewer:
    def __init__(self, samples, previous_manifest, thesis_fit_manifest):
        self.samples = samples
        self.previous_manifest = previous_manifest
        self.thesis_fit_manifest = thesis_fit_manifest
        self.current_index = 0
        self.live_visible = self.dead_visible = self.pith_visible = True
        self.mask_actor_sets = []
        self.active_pipelines = []
        self.row_corners = [None] * 3
        self.scale_labels = []
        self.scale_lines = []
        self.equal_scale = EQUAL_MM_SCALE
        self.tight_crop = TIGHT_DISPLAY_CROP
        self.focus_row = None
        self._camera_guard = False
        self._fitting = False
        self._last_window_size = None
        self.cameras = [vtk.vtkCamera() for _ in range(3)]
        self.renderers = [vtk.vtkRenderer() for _ in range(12)]
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(*WINDOW_SIZE)
        for i, renderer in enumerate(self.renderers):
            renderer.SetBackground(*( (0.95, 0.92, 0.85) if i % 4 in (0, 2) else (0.97, 0.97, 0.92)))
            renderer.SetActiveCamera(self.cameras[i // 4])
            self.render_window.AddRenderer(renderer)
        for row, camera in enumerate(self.cameras):
            camera.ParallelProjectionOn()
            camera.AddObserver('ModifiedEvent', lambda obj, event, r=row: self.camera_changed(r))
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        # Suppress VTK default character shortcuts, which otherwise reset one
        # renderer using padded bounds or conflict with custom layer keys.
        self.style.AddObserver('CharEvent', lambda obj, event: None)
        self.interactor.SetInteractorStyle(self.style)
        self.interactor.AddObserver('KeyPressEvent', self.on_key_press)
        self.interactor.AddObserver('EndInteractionEvent', self.end_interaction)
        self.render_window.AddObserver('StartEvent', self.before_render)
        self.set_layout()

    def visible_rows(self):
        return list(range(3)) if self.focus_row is None else [self.focus_row]

    def set_layout(self):
        rows = self.visible_rows()
        for row in range(3):
            for col in range(4):
                renderer = self.renderers[4 * row + col]
                visible = row in rows
                renderer.SetDraw(visible)
                renderer.SetInteractive(visible)
                if visible:
                    i = rows.index(row)
                    renderer.SetViewport(col / 4, 1 - (i + 1) / len(rows),
                                         (col + 1) / 4, 1 - i / len(rows))
                else:
                    renderer.SetViewport(0, 0, 0, 0)

    def camera_changed(self, row):
        if self._camera_guard or self._fitting:
            return
        self._camera_guard = True
        try:
            if self.equal_scale:
                scale = self.cameras[row].GetParallelScale()
                for camera in self.cameras:
                    camera.SetParallelScale(scale)
            self.update_scales()
        finally:
            self._camera_guard = False

    def end_interaction(self, obj, event):
        for renderer in self.renderers:
            if renderer.GetDraw():
                renderer.ResetCameraClippingRange()
        self.update_scales()
        self.render_window.Render()

    def before_render(self, obj, event):
        size = tuple(self.render_window.GetSize())
        if size != self._last_window_size:
            self._last_window_size = size
            self.fit_cameras(reset_orientation=False)
        else:
            self.update_scales()

    def fit_cameras(self, reset_orientation=False):
        if self._fitting:
            return
        self._fitting = True
        try:
            width, height = self.render_window.GetSize()
            required = {}
            for row in self.visible_rows():
                corners = self.row_corners[row]
                if corners is None:
                    continue
                center = corners.mean(axis=0)
                camera = self.cameras[row]
                direction = np.array(camera.GetDirectionOfProjection())
                up = np.array(camera.GetViewUp())
                if reset_orientation:
                    direction, up = np.array([0., 0., -1.]), np.array([0., 1., 0.])
                direction /= np.linalg.norm(direction)
                right = np.cross(direction, up)
                right /= np.linalg.norm(right)
                up = np.cross(right, direction)
                delta = corners - center
                half_w = max(float(np.max(np.abs(delta @ right))), 0.5)
                half_h = max(float(np.max(np.abs(delta @ up))), 0.5)
                viewport = self.renderers[4 * row].GetViewport()
                aspect = max(1, width * (viewport[2] - viewport[0])) / max(1, height * (viewport[3] - viewport[1]))
                required[row] = max(half_h, half_w / aspect) / FRAME_FILL
                distance = max(float(np.linalg.norm(np.ptp(corners, axis=0))) * 3, 100.)
                camera.SetFocalPoint(*center)
                camera.SetPosition(*(center - direction * distance))
                camera.SetViewUp(*up)
                camera.ParallelProjectionOn()
            common = max(required.values(), default=50.)
            for row, value in required.items():
                self.cameras[row].SetParallelScale(common if self.equal_scale else value)
            for renderer in self.renderers:
                if renderer.GetDraw():
                    renderer.ResetCameraClippingRange()
        finally:
            self._fitting = False
        self.update_scales()

    def add_scale(self, row, renderer):
        line = vtk.vtkLineSource()
        coords = vtk.vtkCoordinate()
        coords.SetCoordinateSystemToNormalizedViewport()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputConnection(line.GetOutputPort())
        mapper.SetTransformCoordinate(coords)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.1, 0.1, 0.1)
        actor.GetProperty().SetLineWidth(3)
        renderer.AddActor2D(actor)
        label = make_text_actor('', 12)
        label.SetPosition(0.04, 0.09)
        renderer.AddActor2D(label)
        self.scale_lines.append((row, renderer, line))
        self.scale_labels.append(label)

    def update_scales(self):
        width, height = self.render_window.GetSize()
        for (row, renderer, line), label in zip(self.scale_lines, self.scale_labels):
            if not renderer.GetDraw():
                continue
            v = renderer.GetViewport()
            aspect = max(1, width * (v[2] - v[0])) / max(1, height * (v[3] - v[1]))
            span = 2 * self.cameras[row].GetParallelScale() * aspect
            target = max(span * 0.20, 1e-9)
            power = 10. ** np.floor(np.log10(target))
            bar_mm = max(a * power for a in (1, 2, 5, 10) if a * power <= target)
            line.SetPoint1(0.04, 0.035, 0)
            line.SetPoint2(0.04 + bar_mm / span, 0.035, 0)
            label.SetInput(f'{bar_mm:g} mm | ' + ('Same mm scale' if self.equal_scale else 'Row fit'))

    def set_layer_visibility(self, layer, visible):
        for actors in self.mask_actor_sets:
            actors[layer].SetVisibility(visible)

    def apply_visibility(self):
        for layer, visible in ((DISPLAY_LIVE_VALUE, self.live_visible),
                               (DISPLAY_DEAD_VALUE, self.dead_visible),
                               (DISPLAY_PITH_VALUE, self.pith_visible)):
            self.set_layer_visibility(layer, visible)

    def add_titles(self, row, name, identity):
        for col, title in enumerate(('Wet image', 'Wet mask', 'Dry image', 'Dry mask')):
            text = f'{name} | {title}'
            self.renderers[4 * row + col].AddActor2D(make_title_actor(text))

    def add_missing_row(self, row, name, identity, reason):
        self.add_titles(row, name, identity)
        for col in range(4):
            self.renderers[4 * row + col].AddActor2D(make_missing_actor(reason))

    def add_loaded_row(self, row, name, identity, loaded, knot_id):
        display = crop_for_display(loaded, knot_id, self.tight_crop)
        self.row_corners[row] = physical_corners(display)
        wet = numpy_volume_to_vtk(display.wet_data, display.geometry)
        dry = numpy_volume_to_vtk(display.dry_data, display.geometry)
        mask = grouped_mask_pipeline(display.mask_data, display.geometry, knot_id)
        wet_mask = create_mask_actor_set(mask)
        dry_mask = create_mask_actor_set(mask)
        self.renderers[4 * row].AddVolume(create_image_volume(wet))
        self.renderers[4 * row + 2].AddVolume(create_image_volume(dry))
        for col, actors in ((1, wet_mask), (3, dry_mask)):
            for label in MASK_RENDER_ORDER:
                self.renderers[4 * row + col].AddVolume(actors[label])
        self.active_pipelines.extend((wet, dry, mask))
        self.mask_actor_sets.extend((wet_mask, dry_mask))
        self.add_titles(row, name, identity)
        for col in range(4):
            self.add_scale(row, self.renderers[4 * row + col])
        print(f'  {name}: full shape {loaded.mask_data.shape}, '
              f'display shape {display.mask_data.shape}, '
              f'spacing {display.geometry.spacing.tolist()} mm')

    def load_outcomes(
        self,
        sample: ComparisonSample,
    ) -> Tuple[ResolutionOutcome, ResolutionOutcome, ResolutionOutcome]:
        key = (sample.tree_number, sample.disk_number, sample.knot_id)

        original = load_resolution_outcome(
            sample.original,
            "Original",
            ORIGINAL_EXPECTED_SPACING_MM,
            None,
        )
        previous = load_resolution_outcome(
            sample.previous_resampling,
            "Previous resampling",
            resampled_expected_spacing(),
            self.previous_manifest.get(key),
        )
        thesis_fit = load_resolution_outcome(
            sample.thesis_fit,
            "Thesis-fit study block",
            None,
            self.thesis_fit_manifest.get(key),
            expected_shape=THESIS_FIT_SHAPE,
            minimum_spacing=resampled_expected_spacing(),
        )

        if original.loaded is not None and previous.loaded is not None:
            error = cross_resolution_error(original.loaded, previous.loaded)
            if error:
                previous = ResolutionOutcome(
                    loaded=None,
                    reason="Files exist but geometry is invalid.\n" + error,
                )

        # The thesis-fit block uses its own aligned local frame, so it is not
        # required to share the original crop's axis directions.
        return original, previous, thesis_fit


    def show_sample(self, index):
        self.current_index = index % len(self.samples)
        sample = self.samples[self.current_index]
        identity = f'Tree {sample.tree_number:02d} | Disk {sample.disk_number:02d} | Knot {sample.knot_id:02d}'
        print(identity)
        outcomes = self.load_outcomes(sample)
        for renderer in self.renderers:
            renderer.RemoveAllViewProps()
        self.mask_actor_sets, self.active_pipelines = [], []
        self.scale_labels, self.scale_lines = [], []
        self.row_corners = [None] * 3
        names = ('Original 0.5 mm', 'Previous 1 x 1 x 10 mm', 'Thesis fit 160 x 80 x 80')
        for row, (name, outcome) in enumerate(zip(names, outcomes)):
            if outcome.loaded is None:
                self.add_missing_row(row, name, identity, outcome.reason)
            else:
                self.add_loaded_row(row, name, identity, outcome.loaded, sample.knot_id)
        self.apply_visibility()
        self.fit_cameras(reset_orientation=True)
        self.render_window.SetWindowName(
            f'{identity} | {self.current_index + 1}/{len(self.samples)} | '
            'Drag: row rotation | M: scale mode | 1/2/3: row | 0: all | R: fit | B: crop')
        self.render_window.Render()

    def show_with_error_report(self, index):
        try:
            self.show_sample(index)
        except Exception as error:
            import traceback
            traceback.print_exc()
            print(f'Could not display knot: {error}')

    def on_key_press(self, obj, event):
        key = obj.GetKeySym().lower()
        if key in ('right', 'down', 'space'):
            self.show_with_error_report(self.current_index + 1)
        elif key in ('left', 'up'):
            self.show_with_error_report(self.current_index - 1)
        elif key in ('home', 'end'):
            self.show_with_error_report(0 if key == 'home' else len(self.samples) - 1)
        elif key in ('l', 'd', 'p'):
            attr = {'l': 'live_visible', 'd': 'dead_visible', 'p': 'pith_visible'}[key]
            setattr(self, attr, not getattr(self, attr))
            self.apply_visibility()
        elif key == 'm':
            self.equal_scale = not self.equal_scale
            self.fit_cameras()
        elif key in ('0', '1', '2', '3'):
            self.focus_row = None if key == '0' else int(key) - 1
            self.set_layout()
            self.fit_cameras()
        elif key == 'b':
            self.tight_crop = not self.tight_crop
            self.show_with_error_report(self.current_index)
        elif key == 'r':
            self.fit_cameras()
        elif key == 'f':
            self.render_window.SetFullScreen(not self.render_window.GetFullScreen())
        elif key in ('escape', 'q'):
            self.render_window.Finalize()
            self.interactor.TerminateApp()
            return
        self.render_window.Render()

    def start(self):
        self.interactor.Initialize()
        if START_FULL_SCREEN:
            self.render_window.SetFullScreen(True)
        self.show_with_error_report(0)
        self.interactor.Start()


def main():
    samples = find_samples()
    if not samples:
        print('No complete original knot crops found. Check ORIGINAL_ROOT.')
        return
    print(__doc__)
    viewer = ThreeResolutionKnotViewer(
        samples, load_manifest_records(PREVIOUS_RESAMPLED_ROOT),
        load_manifest_records(THESIS_FIT_ROOT))
    viewer.start()


def cli(argv=None):
    import argparse
    global ORIGINAL_ROOT, PREVIOUS_RESAMPLED_ROOT, THESIS_FIT_ROOT
    global FIRST_TREE_NUMBER, LAST_TREE_NUMBER, START_FULL_SCREEN
    parser = argparse.ArgumentParser(description='View original, optional previous resampling, and study-fit blocks.')
    parser.add_argument('--original-root', default=ORIGINAL_ROOT)
    parser.add_argument('--previous-root', default=PREVIOUS_RESAMPLED_ROOT)
    parser.add_argument('--block-root', default=THESIS_FIT_ROOT)
    parser.add_argument('--first-tree', type=int, default=FIRST_TREE_NUMBER)
    parser.add_argument('--last-tree', type=int, default=LAST_TREE_NUMBER)
    parser.add_argument('--windowed', action='store_true')
    args = parser.parse_args(argv)
    ORIGINAL_ROOT, PREVIOUS_RESAMPLED_ROOT = args.original_root, args.previous_root
    THESIS_FIT_ROOT = args.block_root
    FIRST_TREE_NUMBER, LAST_TREE_NUMBER = args.first_tree, args.last_tree
    START_FULL_SCREEN = not args.windowed
    paths.require_input_directory(ORIGINAL_ROOT)
    main()


if __name__ == '__main__':
    cli()
