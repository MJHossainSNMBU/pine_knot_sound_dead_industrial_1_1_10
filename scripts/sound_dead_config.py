"""Fixed constants for the published study sound and dead replication."""

from __future__ import annotations

import paths as paths


EXPECTED_TREE_NUMBERS = tuple(range(1, 25))
VALIDATION_TREE_NUMBERS = (5, 8, 20)
TEST_TREE_NUMBERS = (2, 11, 23)
TRAIN_TREE_NUMBERS = tuple(
    tree
    for tree in EXPECTED_TREE_NUMBERS
    if tree not in set(VALIDATION_TREE_NUMBERS + TEST_TREE_NUMBERS)
)

FULL_BLOCK_SHAPE = (160, 80, 80)
PATCH_SHAPE = (11, 80, 80)
PATCH_HALF_WIDTH = PATCH_SHAPE[0] // 2

SOUND_CLASS = 0
DEAD_CLASS = 1
COARSE_RADIAL_STEP = 6

DEFAULT_PROJECT_ROOT = str(paths.PROJECT_ROOT)
DEFAULT_BLOCK_ROOT = str(paths.BLOCK_ROOT)
DEFAULT_DATASET_ROOT = str(paths.DATASET_ROOT)
DEFAULT_INDEX_CSV = str(paths.INDEX_CSV)
DEFAULT_RESULTS_ROOT = str(paths.RESULTS_ROOT)


# Values explicitly reported for the published study evaluation.
PUBLISHED_REFERENCE = {
    "full_block_shape_r_t_z": FULL_BLOCK_SHAPE,
    "classifier_input_shape_r_t_z": PATCH_SHAPE,
    "coarse_radial_step": COARSE_RADIAL_STEP,
    "test_intersections": 577,
    "test_knots": 158,
    "confusion_matrix": {
        "true_sound": 301,
        "sound_predicted_dead": 39,
        "dead_predicted_sound": 39,
        "true_dead": 198,
    },
    "final_status_accuracy": 499.0 / 577.0,
    "sound_recall": 301.0 / 340.0,
    "dead_recall": 198.0 / 237.0,
    "balanced_accuracy": 0.5 * (301.0 / 340.0 + 198.0 / 237.0),
    "milliseconds_per_subvolume": 0.42,
    "mean_evaluations_per_knot": 23.0,
    "milliseconds_per_knot": 10.0,
    "reference_gpu": "NVIDIA RTX 2080",
    "reference_cpu": "Intel i7-4770 3.4 GHz",
}


def validate_fixed_tree_split() -> None:
    """Fail if the fixed 18, 3, 3 tree split is inconsistent."""

    train = set(TRAIN_TREE_NUMBERS)
    validation = set(VALIDATION_TREE_NUMBERS)
    test = set(TEST_TREE_NUMBERS)
    expected = set(EXPECTED_TREE_NUMBERS)

    if train & validation or train & test or validation & test:
        raise RuntimeError("Train, validation, and test trees overlap.")
    if train | validation | test != expected:
        raise RuntimeError("The tree split does not cover trees 1 through 24.")
    if len(TRAIN_TREE_NUMBERS) != 18:
        raise RuntimeError("The training split must contain 18 trees.")


validate_fixed_tree_split()
