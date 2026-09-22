"""Evaluate study geometry classification, transition accuracy, and timing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sound_dead_config import (
    COARSE_RADIAL_STEP,
    DEFAULT_BLOCK_ROOT,
    DEFAULT_INDEX_CSV,
    DEFAULT_RESULTS_ROOT,
    PATCH_SHAPE,
    PUBLISHED_REFERENCE,
    TEST_TREE_NUMBERS,
    TRAIN_TREE_NUMBERS,
    VALIDATION_TREE_NUMBERS,
)
from sound_dead_data import (
    PatchRecord,
    extract_patch,
    group_records_by_knot,
    load_block,
    make_loader,
    normalize_ct_patch,
    read_patch_index,
    split_records_by_fixed_trees,
)
from sound_dead_metrics import (
    binary_metrics,
    boundary_error_metrics,
    fit_monotonic_transition,
)
from sound_dead_model import ModelConfig, SoundDeadCNN


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the held out trees using the study coarse to fine rule."
        )
    )
    parser.add_argument("--block-root", default=DEFAULT_BLOCK_ROOT)
    parser.add_argument("--index-csv", default=DEFAULT_INDEX_CSV)
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(DEFAULT_RESULTS_ROOT, "best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(DEFAULT_RESULTS_ROOT, "evaluation"),
    )
    parser.add_argument("--image-state", choices=("wet", "dry"), default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Dead probability threshold. The study does not publish it, so "
            "0.5 is the fixed primary reconstruction."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--volume-cache-size", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--timing-warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.block_root):
        raise FileNotFoundError(f"Block root does not exist: {args.block_root}")
    if not os.path.isfile(args.index_csv):
        raise FileNotFoundError(f"Patch index does not exist: {args.index_csv}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("Threshold must lie between zero and one.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size or worker count is invalid.")
    if args.volume_cache_size < 1:
        raise ValueError("Volume cache size must be positive.")
    if args.timing_repeats < 1 or args.timing_warmup < 0:
        raise ValueError("Timing repeat or warmup count is invalid.")
    existing = (
        sorted(os.listdir(args.output_dir))
        if os.path.isdir(args.output_dir)
        else []
    )
    if existing:
        raise FileExistsError(
            "This evaluation folder is not empty and will not be overwritten: "
            f"{args.output_dir}. Existing entries: {existing[:10]}. Choose a "
            "new output folder."
        )


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_checkpoint(checkpoint: Mapping, image_state: str) -> None:
    expected_lists = {
        "train_trees": tuple(TRAIN_TREE_NUMBERS),
        "validation_trees": tuple(VALIDATION_TREE_NUMBERS),
        "test_trees": tuple(TEST_TREE_NUMBERS),
        "patch_shape": tuple(PATCH_SHAPE),
    }
    for name, expected in expected_lists.items():
        observed = tuple(checkpoint.get(name, ()))
        if observed != expected:
            raise ValueError(f"Checkpoint {name} is {observed}, expected {expected}.")
    if checkpoint.get("image_state") != image_state:
        raise ValueError(
            f"Checkpoint uses {checkpoint.get('image_state')}, not {image_state}."
        )


def model_from_checkpoint(checkpoint: Mapping, device: torch.device):
    raw = dict(checkpoint.get("model_config", {}))
    config = ModelConfig(
        input_shape=tuple(raw.get("input_shape", PATCH_SHAPE)),
        channels=tuple(raw.get("channels", (16, 32, 64))),
        kernel_size=int(raw.get("kernel_size", 3)),
    )
    model = SoundDeadCNN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled)


def json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def write_json(path: str, value) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(json_ready(value), output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_patch_classifier(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
) -> Tuple[dict, List[dict]]:
    criterion = torch.nn.BCEWithLogitsLoss()
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    rows: List[dict] = []
    loss_sum = 0.0
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)

        labels_cpu = labels.cpu().to(torch.int64).tolist()
        probabilities_cpu = probabilities.cpu().tolist()
        count = len(labels_cpu)
        loss_sum += float(loss.item()) * count
        sample_count += count
        labels_all.extend(labels_cpu)
        probabilities_all.extend(probabilities_cpu)

        for index in range(count):
            probability = float(probabilities_cpu[index])
            prediction = int(probability >= threshold)
            actual = int(labels_cpu[index])
            rows.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "tree": int(batch["tree"][index]),
                    "disk": int(batch["disk"][index]),
                    "knot_id": int(batch["knot_id"][index]),
                    "center_r": int(batch["center_r"][index]),
                    "radial_mm_from_pith": float(
                        batch["radial_mm_from_pith"][index]
                    ),
                    "actual_label": actual,
                    "actual_name": "sound" if actual == 0 else "dead",
                    "dead_probability": probability,
                    "predicted_label": prediction,
                    "predicted_name": "sound" if prediction == 0 else "dead",
                    "correct": int(prediction == actual),
                }
            )

    metrics = binary_metrics(
        labels_all,
        probabilities_all,
        threshold=threshold,
        loss=loss_sum / max(1, sample_count),
    )
    return metrics, rows


def metrics_by_tree(
    rows: Sequence[dict],
    *,
    actual_key: str,
    probability_key: str,
    threshold: float,
) -> List[dict]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["tree"])].append(row)
    output = []
    for tree in sorted(grouped):
        values = grouped[tree]
        metrics = binary_metrics(
            [int(row[actual_key]) for row in values],
            [float(row[probability_key]) for row in values],
            threshold=threshold,
        )
        output.append({"tree": tree, **metrics})
    return output


def select_coarse_centres(valid_centres: Sequence[int]) -> List[int]:
    centres = sorted({int(value) for value in valid_centres})
    if not centres:
        raise ValueError("A knot has no valid radial centres.")
    coarse = [
        center
        for center in centres
        if (center - centres[0]) % COARSE_RADIAL_STEP == 0
    ]
    if not coarse:
        coarse = [centres[0]]
    if centres[-1] not in coarse:
        coarse.append(centres[-1])
    return coarse


@torch.no_grad()
def predict_one_center(
    model: torch.nn.Module,
    block: np.ndarray,
    record: PatchRecord,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    patch = np.array(
        extract_patch(block, record.start_r, record.stop_r),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    patch = normalize_ct_patch(patch)
    inputs = torch.from_numpy(patch[None, None, ...]).to(
        device,
        non_blocking=True,
    )
    with autocast(device, amp_enabled):
        logit = model(inputs)
    return float(torch.sigmoid(logit)[0].item())


@torch.no_grad()
def infer_knot_transition(
    *,
    model: torch.nn.Module,
    block: np.ndarray,
    records: Sequence[PatchRecord],
    threshold: float,
    device: torch.device,
    amp_enabled: bool,
) -> dict:
    records_by_center = {record.center_r: record for record in records}
    valid_centres = sorted(records_by_center)
    if len(records_by_center) != len(records):
        raise ValueError("Duplicate centre in one knot.")
    coarse_centres = select_coarse_centres(valid_centres)
    probabilities: Dict[int, float] = {}
    for center in coarse_centres:
        probabilities[center] = predict_one_center(
            model,
            block,
            records_by_center[center],
            device,
            amp_enabled,
        )
    coarse_predictions = [
        int(probabilities[center] >= threshold) for center in coarse_centres
    ]
    sound_only_boundary = valid_centres[-1] + 1
    _, coarse_split_index, coarse_disagreements = fit_monotonic_transition(
        coarse_centres,
        coarse_predictions,
        sound_only_boundary=sound_only_boundary,
    )

    fine_centres: List[int] = []
    if 0 < coarse_split_index < len(coarse_centres):
        left = coarse_centres[coarse_split_index - 1]
        right = coarse_centres[coarse_split_index]
        fine_centres = [
            center
            for center in valid_centres
            if left < center < right and center not in probabilities
        ]
        for center in fine_centres:
            probabilities[center] = predict_one_center(
                model,
                block,
                records_by_center[center],
                device,
                amp_enabled,
            )

    evaluated_centres = sorted(probabilities)
    evaluated_predictions = [
        int(probabilities[center] >= threshold) for center in evaluated_centres
    ]
    boundary, split_index, disagreements = fit_monotonic_transition(
        evaluated_centres,
        evaluated_predictions,
        sound_only_boundary=sound_only_boundary,
    )
    return {
        "predicted_boundary_r": boundary,
        "predicted_globally_sound": int(boundary > valid_centres[-1]),
        "valid_center_minimum": valid_centres[0],
        "valid_center_maximum": valid_centres[-1],
        "coarse_split_index": coarse_split_index,
        "coarse_disagreements": coarse_disagreements,
        "final_split_index": split_index,
        "final_disagreements": disagreements,
        "coarse_centres": coarse_centres,
        "coarse_probabilities": [probabilities[value] for value in coarse_centres],
        "fine_centres": fine_centres,
        "fine_probabilities": [probabilities[value] for value in fine_centres],
        "evaluated_patch_count": len(evaluated_centres),
    }


def warm_up_timing(
    model: torch.nn.Module,
    block: np.ndarray,
    record: PatchRecord,
    device: torch.device,
    amp_enabled: bool,
    repetitions: int,
) -> None:
    for _ in range(repetitions):
        predict_one_center(model, block, record, device, amp_enabled)
    sync_device(device)


def evaluate_transitions_and_timing(
    *,
    model: torch.nn.Module,
    test_records: Sequence[PatchRecord],
    block_root: str,
    image_state: str,
    threshold: float,
    device: torch.device,
    amp_enabled: bool,
    timing_repeats: int,
    timing_warmup: int,
) -> Tuple[dict, List[dict], List[dict], List[dict]]:
    grouped = group_records_by_knot(test_records)
    knot_rows: List[dict] = []
    status_rows: List[dict] = []
    timing_rows: List[dict] = []
    warmed_up = False

    for knot_index, (key, records) in enumerate(grouped.items(), start=1):
        first = records[0]
        relative_path = first.image_relpath(image_state)
        block = load_block(block_root, relative_path)
        if not warmed_up:
            warm_up_timing(
                model,
                block,
                first,
                device,
                amp_enabled,
                timing_warmup,
            )
            warmed_up = True

        elapsed_values: List[float] = []
        predictions: List[dict] = []
        for _ in range(timing_repeats):
            sync_device(device)
            started = time.perf_counter()
            result = infer_knot_transition(
                model=model,
                block=block,
                records=records,
                threshold=threshold,
                device=device,
                amp_enabled=amp_enabled,
            )
            sync_device(device)
            elapsed_values.append((time.perf_counter() - started) * 1000.0)
            predictions.append(result)

        prediction = predictions[0]
        predicted_boundaries = {
            int(value["predicted_boundary_r"]) for value in predictions
        }
        evaluated_counts = {
            int(value["evaluated_patch_count"]) for value in predictions
        }
        if len(predicted_boundaries) != 1 or len(evaluated_counts) != 1:
            raise RuntimeError(f"Nondeterministic transition output for knot {key}.")

        predicted_boundary = int(prediction["predicted_boundary_r"])
        actual_boundary = int(first.transition_r)
        spacing = float(first.spacing_radial_mm)
        error_voxels = predicted_boundary - actual_boundary
        error_mm = float(error_voxels * spacing)
        actual_globally_sound = int(all(record.label == 0 for record in records))
        median_ms = float(statistics.median(elapsed_values))
        mean_ms = float(statistics.fmean(elapsed_values))
        evaluated_count = int(prediction["evaluated_patch_count"])

        timing_row = {
            "tree": first.tree_number,
            "disk": first.disk_number,
            "knot_id": first.knot_id,
            "timing_repeats": timing_repeats,
            "evaluated_patch_count": evaluated_count,
            "median_ms_per_knot": median_ms,
            "mean_ms_per_knot": mean_ms,
            "minimum_ms_per_knot": min(elapsed_values),
            "maximum_ms_per_knot": max(elapsed_values),
            "median_ms_per_evaluated_patch": median_ms / evaluated_count,
        }
        timing_rows.append(timing_row)

        knot_rows.append(
            {
                "tree": first.tree_number,
                "disk": first.disk_number,
                "knot_id": first.knot_id,
                "actual_boundary_r": actual_boundary,
                "predicted_boundary_r": predicted_boundary,
                "boundary_error_voxels": error_voxels,
                "absolute_error_voxels": abs(error_voxels),
                "spacing_radial_mm": spacing,
                "actual_transition_mm_from_pith": first.transition_mm_from_pith,
                "predicted_transition_mm_from_pith": (
                    first.transition_mm_from_pith + error_mm
                ),
                "boundary_error_mm": error_mm,
                "absolute_error_mm": abs(error_mm),
                "actual_globally_sound": actual_globally_sound,
                **prediction,
                "coarse_centres": json.dumps(prediction["coarse_centres"]),
                "coarse_probabilities": json.dumps(
                    prediction["coarse_probabilities"]
                ),
                "fine_centres": json.dumps(prediction["fine_centres"]),
                "fine_probabilities": json.dumps(
                    prediction["fine_probabilities"]
                ),
                "median_processing_ms": median_ms,
                "mean_processing_ms": mean_ms,
                "timing_repeats": timing_repeats,
            }
        )

        for record in records:
            predicted_label = int(record.center_r >= predicted_boundary)
            status_rows.append(
                {
                    "sample_id": record.sample_id,
                    "tree": record.tree_number,
                    "disk": record.disk_number,
                    "knot_id": record.knot_id,
                    "center_r": record.center_r,
                    "radial_mm_from_pith": record.radial_mm_from_pith,
                    "actual_label": record.label,
                    "actual_name": "sound" if record.label == 0 else "dead",
                    "predicted_label": predicted_label,
                    "predicted_name": (
                        "sound" if predicted_label == 0 else "dead"
                    ),
                    "correct": int(predicted_label == record.label),
                    "actual_boundary_r": actual_boundary,
                    "predicted_boundary_r": predicted_boundary,
                }
            )

        print(
            f"{knot_index} of {len(grouped)} | Tree {key[0]:02d} | "
            f"Disk {key[1]:02d} | Knot {key[2]:02d} | "
            f"{evaluated_count} evaluations | {median_ms:.3f} ms"
        )

    status_metrics = binary_metrics(
        [int(row["actual_label"]) for row in status_rows],
        [float(row["predicted_label"]) for row in status_rows],
        threshold=0.5,
    )
    dead_transition_errors = [
        float(row["boundary_error_mm"])
        for row in knot_rows
        if int(row["actual_globally_sound"]) == 0
    ]
    all_errors = [float(row["boundary_error_mm"]) for row in knot_rows]
    boundary_dead = (
        boundary_error_metrics(dead_transition_errors)
        if dead_transition_errors
        else {}
    )
    boundary_all = boundary_error_metrics(all_errors)

    total_median_ms = sum(float(row["median_ms_per_knot"]) for row in timing_rows)
    total_evaluations = sum(int(row["evaluated_patch_count"]) for row in timing_rows)
    timing_summary = {
        "knots": len(timing_rows),
        "timing_repeats_per_knot": timing_repeats,
        "warmup_single_patch_inferences": timing_warmup,
        "mean_evaluations_per_knot": float(
            np.mean([row["evaluated_patch_count"] for row in timing_rows])
        ),
        "median_evaluations_per_knot": float(
            np.median([row["evaluated_patch_count"] for row in timing_rows])
        ),
        "mean_ms_per_knot": float(
            np.mean([row["median_ms_per_knot"] for row in timing_rows])
        ),
        "median_ms_per_knot": float(
            np.median([row["median_ms_per_knot"] for row in timing_rows])
        ),
        "weighted_ms_per_evaluated_patch": float(
            total_median_ms / max(1, total_evaluations)
        ),
        "timed_scope": (
            "NHDR block already loaded. Includes patch extraction, nonzero z "
            "score, host to device transfer, model inference, sigmoid, and "
            "coarse to fine decision logic. Uses sequential batch size one."
        ),
    }
    transition_metrics = {
        "status_at_every_indexed_test_position": status_metrics,
        "boundary_dead_transition_knots": boundary_dead,
        "boundary_all_knots_surrogate": boundary_all,
        "globally_sound_knots": {
            "actual": sum(int(row["actual_globally_sound"]) for row in knot_rows),
            "predicted": sum(
                int(row["predicted_globally_sound"]) for row in knot_rows
            ),
            "correct": sum(
                int(row["actual_globally_sound"])
                == int(row["predicted_globally_sound"])
                for row in knot_rows
            ),
            "total": len(knot_rows),
        },
        "timing": timing_summary,
    }
    return transition_metrics, knot_rows, status_rows, timing_rows


def plot_confusion_matrix(metrics: Mapping, title: str, output_path: str) -> None:
    matrix = np.asarray(
        [
            [metrics["true_sound"], metrics["sound_predicted_dead"]],
            [metrics["dead_predicted_sound"], metrics["true_dead"]],
        ],
        dtype=np.int64,
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    fractions = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks([0, 1], labels=["Sound", "Dead"])
    axis.set_yticks([0, 1], labels=["Sound", "Dead"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    cutoff = float(matrix.max()) * 0.5
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > cutoff else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{fractions[row, column] * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_boundary_errors(knot_rows: Sequence[dict], output_path: str) -> None:
    errors = [
        float(row["boundary_error_mm"])
        for row in knot_rows
        if int(row["actual_globally_sound"]) == 0
    ]
    if not errors:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(errors, bins=25, color="#2f6fa5", edgecolor="white")
    axis.axvline(0.0, color="black", linewidth=1.5)
    axis.set_title("Study style boundary errors on held out test trees")
    axis.set_xlabel("Predicted minus surrogate boundary in mm")
    axis.set_ylabel("Knot count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def hardware_information(device: torch.device, amp_enabled: bool) -> dict:
    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "not used"
    )
    return {
        "device": str(device),
        "gpu": gpu_name,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "amp_enabled": amp_enabled,
    }


def comparison_rows(
    patch_metrics: Mapping,
    transition_metrics: Mapping,
) -> List[dict]:
    published = PUBLISHED_REFERENCE
    status = transition_metrics["status_at_every_indexed_test_position"]
    boundary = transition_metrics["boundary_dead_transition_knots"]
    timing = transition_metrics["timing"]
    values = [
        (
            "Patch classifier accuracy",
            "not reported",
            patch_metrics["accuracy"],
            "Our unit is every indexed 11 by 80 by 80 test patch.",
        ),
        (
            "Final status accuracy",
            published["final_status_accuracy"],
            status["accuracy"],
            "Study unit is 577 measured intersections. Our unit is every indexed radial test position.",
        ),
        (
            "Final balanced accuracy",
            published["balanced_accuracy"],
            status["balanced_accuracy"],
            "Calculated from each confusion matrix.",
        ),
        (
            "Sound recall",
            published["sound_recall"],
            status["sound_recall"],
            "Sound is class zero.",
        ),
        (
            "Dead recall",
            published["dead_recall"],
            status["dead_recall"],
            "Dead is class one.",
        ),
        (
            "Boundary MAE mm",
            "not reported",
            boundary.get("mae_mm", "not available"),
            "Our reference is the fitted mask surrogate, not a board measurement.",
        ),
        (
            "Boundary RMSE mm",
            "not reported",
            boundary.get("rmse_mm", "not available"),
            "Our reference is the fitted mask surrogate, not a board measurement.",
        ),
        (
            "Mean evaluations per knot",
            published["mean_evaluations_per_knot"],
            timing["mean_evaluations_per_knot"],
            "Both use coarse stride six followed by fine refinement.",
        ),
        (
            "Milliseconds per evaluated patch",
            published["milliseconds_per_subvolume"],
            timing["weighted_ms_per_evaluated_patch"],
            "Hardware and timed scope differ.",
        ),
        (
            "Milliseconds per knot",
            published["milliseconds_per_knot"],
            timing["mean_ms_per_knot"],
            "Hardware and timed scope differ.",
        ),
    ]
    return [
        {
            "metric": metric,
            "published": reference_value,
            "our_fixed_tree_replication": our_value,
            "comparison_note": note,
        }
        for metric, reference_value, our_value, note in values
    ]


def write_text_report(
    path: str,
    patch_metrics: Mapping,
    transition_metrics: Mapping,
    hardware: Mapping,
) -> None:
    status = transition_metrics["status_at_every_indexed_test_position"]
    boundary = transition_metrics["boundary_dead_transition_knots"]
    timing = transition_metrics["timing"]
    study = PUBLISHED_REFERENCE
    lines = [
        "PUBLISHED STUDY SOUND AND DEAD REPLICATION",
        "",
        "Held out test trees: 2, 11, 23",
        "Input: 11 by 80 by 80 from 160 by 80 by 80 thesis fit blocks",
        "",
        "PATCH CLASSIFIER",
        f"Accuracy: {100.0 * float(patch_metrics['accuracy']):.2f}%",
        f"Balanced accuracy: {100.0 * float(patch_metrics['balanced_accuracy']):.2f}%",
        f"ROC AUC: {float(patch_metrics['roc_auc']):.4f}",
        "Published study patch accuracy: not reported",
        "",
        "COARSE TO FINE FINAL STATUS",
        f"Our accuracy: {100.0 * float(status['accuracy']):.2f}%",
        f"Our balanced accuracy: {100.0 * float(status['balanced_accuracy']):.2f}%",
        f"Published study accuracy: {100.0 * float(study['final_status_accuracy']):.2f}%",
        "Published confusion matrix: 301, 39, 39, 198",
        (
            "Our confusion matrix: "
            f"{status['true_sound']}, {status['sound_predicted_dead']}, "
            f"{status['dead_predicted_sound']}, {status['true_dead']}"
        ),
        "",
        "BOUNDARY ERROR ON KNOTS WITH A DEAD TRANSITION",
        (
            f"MAE: {float(boundary['mae_mm']):.3f} mm"
            if boundary
            else "MAE: not available"
        ),
        (
            f"RMSE: {float(boundary['rmse_mm']):.3f} mm"
            if boundary
            else "RMSE: not available"
        ),
        "Published study MAE and RMSE: not reported",
        "",
        "PROCESSING TIME",
        f"Our mean evaluations per knot: {float(timing['mean_evaluations_per_knot']):.2f}",
        f"Our weighted time per evaluation: {float(timing['weighted_ms_per_evaluated_patch']):.3f} ms",
        f"Our mean time per knot: {float(timing['mean_ms_per_knot']):.3f} ms",
        "Published study: 23 evaluations per knot",
        "Published study: 0.42 ms per subvolume and 10 ms per knot",
        f"Our GPU: {hardware['gpu']}",
        f"Study GPU: {study['reference_gpu']}",
        "",
        "COMPARISON LIMITS",
        (
            "The study used physical board measurements and 577 measured test "
            "intersections. This replication uses mask derived surrogate labels "
            "at every indexed radial position and completely held out trees."
        ),
        (
            "The study does not publish the classifier channels, kernels, batch "
            "size, learning rate, normalization, or decision threshold. These "
            "are documented reconstructed choices in experiment_config.json."
        ),
        (
            "Timing is not a direct hardware benchmark. Our timed scope includes "
            "patch extraction, normalization, transfer, inference, and decision logic."
        ),
    ]
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    image_state = args.image_state or str(checkpoint.get("image_state", "wet"))
    amp_enabled = device.type == "cuda" and not args.no_amp
    validate_checkpoint(checkpoint, image_state)
    model = model_from_checkpoint(checkpoint, device)

    records = read_patch_index(args.index_csv)
    _, _, test_records = split_records_by_fixed_trees(records)
    test_loader = make_loader(
        test_records,
        block_root=args.block_root,
        image_state=image_state,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        volume_cache_size=args.volume_cache_size,
        training=False,
        seed=args.seed,
    )

    print("Evaluating every indexed test patch")
    patch_metrics, patch_rows = evaluate_patch_classifier(
        model,
        test_loader,
        device,
        amp_enabled,
        args.threshold,
    )
    patch_tree_rows = metrics_by_tree(
        patch_rows,
        actual_key="actual_label",
        probability_key="dead_probability",
        threshold=args.threshold,
    )

    print("Evaluating study style coarse to fine transitions and timing")
    transition_metrics, knot_rows, status_rows, timing_rows = (
        evaluate_transitions_and_timing(
            model=model,
            test_records=test_records,
            block_root=args.block_root,
            image_state=image_state,
            threshold=args.threshold,
            device=device,
            amp_enabled=amp_enabled,
            timing_repeats=args.timing_repeats,
            timing_warmup=args.timing_warmup,
        )
    )
    status_tree_rows = metrics_by_tree(
        status_rows,
        actual_key="actual_label",
        probability_key="predicted_label",
        threshold=0.5,
    )
    hardware = hardware_information(device, amp_enabled)
    comparison = comparison_rows(patch_metrics, transition_metrics)

    write_json(
        os.path.join(args.output_dir, "test_patch_metrics.json"),
        patch_metrics,
    )
    write_json(
        os.path.join(args.output_dir, "transition_metrics.json"),
        transition_metrics,
    )
    write_json(
        os.path.join(args.output_dir, "timing_summary.json"),
        {"hardware": hardware, **transition_metrics["timing"]},
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_state": image_state,
        "threshold": args.threshold,
        "test_trees": TEST_TREE_NUMBERS,
        "checkpoint": os.path.abspath(args.checkpoint),
        "block_root": os.path.abspath(args.block_root),
        "index_csv": os.path.abspath(args.index_csv),
        "patch_metrics": patch_metrics,
        "transition_metrics": transition_metrics,
        "published_reference": PUBLISHED_REFERENCE,
        "hardware": hardware,
        "comparison_limit": (
            "Study results use physical measured intersections. Replication "
            "results use mask derived labels at indexed positions and held out trees."
        ),
    }
    write_json(
        os.path.join(args.output_dir, "evaluation_summary.json"),
        summary,
    )

    write_csv(
        os.path.join(args.output_dir, "test_patch_predictions.csv"),
        patch_rows,
        list(patch_rows[0]),
    )
    write_csv(
        os.path.join(args.output_dir, "test_patch_metrics_by_tree.csv"),
        patch_tree_rows,
        list(patch_tree_rows[0]),
    )
    write_csv(
        os.path.join(args.output_dir, "knot_transition_predictions.csv"),
        knot_rows,
        list(knot_rows[0]),
    )
    write_csv(
        os.path.join(args.output_dir, "transition_status_predictions.csv"),
        status_rows,
        list(status_rows[0]),
    )
    write_csv(
        os.path.join(
            args.output_dir,
            "transition_status_metrics_by_tree.csv",
        ),
        status_tree_rows,
        list(status_tree_rows[0]),
    )
    write_csv(
        os.path.join(args.output_dir, "timing_by_knot.csv"),
        timing_rows,
        list(timing_rows[0]),
    )
    write_csv(
        os.path.join(args.output_dir, "comparison_with_published.csv"),
        comparison,
        list(comparison[0]),
    )

    plot_confusion_matrix(
        patch_metrics,
        "Patch classifier on held out test trees",
        os.path.join(args.output_dir, "patch_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        transition_metrics["status_at_every_indexed_test_position"],
        "Study style coarse to fine status on test trees",
        os.path.join(
            args.output_dir,
            "transition_status_confusion_matrix.png",
        ),
    )
    plot_boundary_errors(
        knot_rows,
        os.path.join(args.output_dir, "boundary_error_histogram.png"),
    )
    write_text_report(
        os.path.join(args.output_dir, "evaluation_report.txt"),
        patch_metrics,
        transition_metrics,
        hardware,
    )

    status = transition_metrics["status_at_every_indexed_test_position"]
    timing = transition_metrics["timing"]
    print("\nFinished")
    print(f"Patch accuracy: {100.0 * float(patch_metrics['accuracy']):.2f}%")
    print(f"Final status accuracy: {100.0 * float(status['accuracy']):.2f}%")
    print(
        "Published final status accuracy: "
        f"{100.0 * PUBLISHED_REFERENCE['final_status_accuracy']:.2f}%"
    )
    print(f"Mean evaluations per knot: {timing['mean_evaluations_per_knot']:.2f}")
    print(f"Mean processing time per knot: {timing['mean_ms_per_knot']:.3f} ms")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
