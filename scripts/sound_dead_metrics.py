"""Metrics and monotonic transition fitting for study replication."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def binary_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    positive_count = int(np.count_nonzero(labels_array == 1))
    negative_count = int(np.count_nonzero(labels_array == 0))
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end

    positive_rank_sum = float(np.sum(ranks[labels_array == 1]))
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
    loss: float | None = None,
) -> Dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels_array.ndim != 1 or scores.shape != labels_array.shape:
        raise ValueError("Labels and probabilities must be matching 1D arrays.")
    if labels_array.size == 0:
        raise ValueError("At least one prediction is required.")
    if set(labels_array.tolist()) - {0, 1}:
        raise ValueError("Labels must contain only zero and one.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must lie between zero and one.")

    predictions = (scores >= threshold).astype(np.int64)
    true_sound = int(np.count_nonzero((labels_array == 0) & (predictions == 0)))
    sound_predicted_dead = int(
        np.count_nonzero((labels_array == 0) & (predictions == 1))
    )
    dead_predicted_sound = int(
        np.count_nonzero((labels_array == 1) & (predictions == 0))
    )
    true_dead = int(np.count_nonzero((labels_array == 1) & (predictions == 1)))

    accuracy = safe_ratio(true_sound + true_dead, labels_array.size)
    sound_recall = safe_ratio(true_sound, true_sound + sound_predicted_dead)
    dead_recall = safe_ratio(true_dead, true_dead + dead_predicted_sound)
    dead_precision = safe_ratio(true_dead, true_dead + sound_predicted_dead)
    sound_precision = safe_ratio(true_sound, true_sound + dead_predicted_sound)
    f1_dead = safe_ratio(
        2.0 * dead_precision * dead_recall,
        dead_precision + dead_recall,
    )
    f1_sound = safe_ratio(
        2.0 * sound_precision * sound_recall,
        sound_precision + sound_recall,
    )

    result: Dict[str, float | int] = {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.nanmean([sound_recall, dead_recall])),
        "sound_precision": sound_precision,
        "sound_recall": sound_recall,
        "sound_f1": f1_sound,
        "dead_precision": dead_precision,
        "dead_recall": dead_recall,
        "dead_f1": f1_dead,
        "roc_auc": binary_auc(labels_array, scores),
        "true_sound": true_sound,
        "sound_predicted_dead": sound_predicted_dead,
        "dead_predicted_sound": dead_predicted_sound,
        "true_dead": true_dead,
        "samples": int(labels_array.size),
        "threshold": float(threshold),
    }
    if loss is not None:
        result["loss"] = float(loss)
    return result


def fit_monotonic_transition(
    centres: Sequence[int],
    predictions: Sequence[int],
    *,
    sound_only_boundary: int,
) -> Tuple[int, int, int]:
    """Fit one sound to dead transition to possibly noisy predictions.

    The selected split balances dead predictions to its left against sound
    predictions to its right. Its index equals the total count of sound
    predictions. This is the stable transition rule described in the thesis.
    """

    centres_array = np.asarray(centres, dtype=np.int64)
    values = np.asarray(predictions, dtype=np.int64)
    if centres_array.ndim != 1 or values.shape != centres_array.shape:
        raise ValueError("Centres and predictions must be matching 1D arrays.")
    if centres_array.size == 0:
        raise ValueError("At least one evaluated centre is required.")
    if np.any(np.diff(centres_array) <= 0):
        raise ValueError("Centres must be strictly increasing.")
    if set(values.tolist()) - {0, 1}:
        raise ValueError("Predictions must contain only zero and one.")

    split_index = int(np.count_nonzero(values == 0))
    dead_on_left = int(np.count_nonzero(values[:split_index] == 1))
    sound_on_right = int(np.count_nonzero(values[split_index:] == 0))
    if dead_on_left != sound_on_right:
        raise RuntimeError("Stable transition balance invariant failed.")
    disagreements = dead_on_left + sound_on_right
    boundary = (
        int(sound_only_boundary)
        if split_index == centres_array.size
        else int(centres_array[split_index])
    )
    return boundary, split_index, disagreements


def boundary_error_metrics(errors_mm: Sequence[float]) -> Dict[str, float | int]:
    errors = np.asarray(errors_mm, dtype=np.float64)
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError("At least one boundary error is required.")
    if not np.all(np.isfinite(errors)):
        raise ValueError("Boundary errors must be finite.")
    absolute = np.abs(errors)
    return {
        "knots": int(errors.size),
        "mean_error_mm": float(np.mean(errors)),
        "standard_deviation_error_mm": float(np.std(errors)),
        "median_error_mm": float(np.median(errors)),
        "mae_mm": float(np.mean(absolute)),
        "median_absolute_error_mm": float(np.median(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "absolute_error_90th_percentile_mm": float(np.percentile(absolute, 90)),
        "absolute_error_95th_percentile_mm": float(np.percentile(absolute, 95)),
        "within_5_mm_fraction": float(np.mean(absolute <= 5.0)),
        "within_10_mm_fraction": float(np.mean(absolute <= 10.0)),
    }
