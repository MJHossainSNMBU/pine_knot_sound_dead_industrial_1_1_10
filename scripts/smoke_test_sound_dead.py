"""Fast structural checks that do not read the real Pine data."""

from __future__ import annotations

import torch

from sound_dead_config import PATCH_SHAPE, TEST_TREE_NUMBERS
from sound_dead_metrics import binary_metrics, fit_monotonic_transition
from sound_dead_model import SoundDeadCNN


def main() -> None:
    model = SoundDeadCNN()
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1, *PATCH_SHAPE))
    if tuple(output.shape) != (1,):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")

    boundary, split_index, disagreements = fit_monotonic_transition(
        [5, 11, 17, 23],
        [0, 0, 1, 1],
        sound_only_boundary=24,
    )
    if (boundary, split_index, disagreements) != (17, 2, 0):
        raise RuntimeError("Monotonic transition check failed.")

    metrics = binary_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
    )
    if float(metrics["accuracy"]) != 1.0:
        raise RuntimeError("Binary metric check failed.")

    print("Study replication smoke test passed")
    print(f"Input shape: {PATCH_SHAPE}")
    print(f"Held out test trees: {TEST_TREE_NUMBERS}")
    print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
