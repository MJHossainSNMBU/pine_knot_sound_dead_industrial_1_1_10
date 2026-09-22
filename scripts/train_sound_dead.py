"""Train the closest reproducible study style sound and dead classifier."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Mapping

import numpy as np
import torch

from sound_dead_config import (
    DEFAULT_BLOCK_ROOT,
    DEFAULT_INDEX_CSV,
    DEFAULT_RESULTS_ROOT,
    PATCH_SHAPE,
    TEST_TREE_NUMBERS,
    TRAIN_TREE_NUMBERS,
    VALIDATION_TREE_NUMBERS,
)
from sound_dead_data import (
    balance_binary_training_records,
    class_counts,
    make_loader,
    read_patch_index,
    split_records_by_fixed_trees,
    unique_knot_count,
    write_split_manifest,
)
from sound_dead_metrics import binary_metrics
from sound_dead_model import (
    ModelConfig,
    SoundDeadCNN,
    count_parameters,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the study geometry classifier on the fixed Pine tree split."
        )
    )
    parser.add_argument("--block-root", default=DEFAULT_BLOCK_ROOT)
    parser.add_argument("--index-csv", default=DEFAULT_INDEX_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--image-state", choices=("wet", "dry"), default="wet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--volume-cache-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--resume",
        default="",
        help="Path to last_checkpoint.pt from the same output folder.",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.block_root):
        raise FileNotFoundError(f"Block root does not exist: {args.block_root}")
    if not os.path.isfile(args.index_csv):
        raise FileNotFoundError(f"Patch index does not exist: {args.index_csv}")
    if args.epochs < 1 or args.early_stop_patience < 1 or args.lr_patience < 1:
        raise ValueError("Epoch and patience values must be positive.")
    if args.batch_size < 1 or args.evaluation_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Learning rate or weight decay is invalid.")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("lr_factor must lie between zero and one.")
    if args.num_workers < 0 or args.volume_cache_size < 1:
        raise ValueError("Worker count or volume cache size is invalid.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("Threshold must lie between zero and one.")
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")

    existing = (
        sorted(os.listdir(args.output_dir))
        if os.path.isdir(args.output_dir)
        else []
    )
    if existing and not args.resume:
        raise FileExistsError(
            "The output folder is not empty and will not be overwritten: "
            f"{args.output_dir}. Existing entries: {existing[:10]}. Set a new "
            "output folder or use "
            "--resume with last_checkpoint.pt."
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


def initialize_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"study_sound_dead_{os.path.abspath(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "training.log"),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


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


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled)


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
) -> Dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    labels_all: List[int] = []
    probabilities_all: List[float] = []

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            probabilities = torch.sigmoid(logits.detach())
            count = int(labels.numel())
            total_loss += float(loss.item()) * count
            total_samples += count
            labels_all.extend(labels.detach().cpu().to(torch.int64).tolist())
            probabilities_all.extend(probabilities.cpu().tolist())

    return binary_metrics(
        labels_all,
        probabilities_all,
        threshold=threshold,
        loss=total_loss / max(1, total_samples),
    )


def checkpoint_payload(
    *,
    epoch: int,
    model: SoundDeadCNN,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_validation_loss: float,
    epochs_without_improvement: int,
    args: argparse.Namespace,
    validation_metrics: Dict[str, float | int],
) -> dict:
    return {
        "schema_version": 1,
        "replication_target": "published_sound_dead_classifier",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_config": model.config.to_dict(),
        "best_validation_loss": float(best_validation_loss),
        "epochs_without_improvement": int(epochs_without_improvement),
        "image_state": args.image_state,
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "train_trees": tuple(TRAIN_TREE_NUMBERS),
        "validation_trees": tuple(VALIDATION_TREE_NUMBERS),
        "test_trees": tuple(TEST_TREE_NUMBERS),
        "patch_shape": tuple(PATCH_SHAPE),
        "validation_metrics": validation_metrics,
    }


def atomic_torch_save(payload: dict, path: str) -> None:
    temporary_path = path + ".partial"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def append_history(path: str, row: dict) -> None:
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_resume_checkpoint(checkpoint: Mapping, args: argparse.Namespace) -> None:
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
    if checkpoint.get("image_state") != args.image_state:
        raise ValueError("Checkpoint image state differs from this run.")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    set_global_seed(args.seed)
    logger = initialize_logger(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp

    records = read_patch_index(args.index_csv)
    train_raw, validation_records, test_records = split_records_by_fixed_trees(
        records
    )
    train_records = balance_binary_training_records(train_raw, args.seed)

    write_split_manifest(
        os.path.join(args.output_dir, "split_manifest.csv"),
        {
            "train_before_balance": train_raw,
            "train_selected_balanced": train_records,
            "validation": validation_records,
            "test_untouched": test_records,
        },
    )

    train_loader = make_loader(
        train_records,
        block_root=args.block_root,
        image_state=args.image_state,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        volume_cache_size=args.volume_cache_size,
        training=True,
        seed=args.seed,
    )
    validation_loader = make_loader(
        validation_records,
        block_root=args.block_root,
        image_state=args.image_state,
        batch_size=args.evaluation_batch_size,
        num_workers=args.num_workers,
        volume_cache_size=args.volume_cache_size,
        training=False,
        seed=args.seed,
    )

    model = SoundDeadCNN(ModelConfig()).to(device)
    parameter_total, parameter_trainable = count_parameters(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = make_grad_scaler(amp_enabled)

    experiment_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "block_root": os.path.abspath(args.block_root),
        "index_csv": os.path.abspath(args.index_csv),
        "output_dir": os.path.abspath(args.output_dir),
        "image_state": args.image_state,
        "fixed_tree_split": {
            "train": TRAIN_TREE_NUMBERS,
            "validation": VALIDATION_TREE_NUMBERS,
            "test": TEST_TREE_NUMBERS,
        },
        "patch_counts": {
            "train_before_balance": class_counts(train_raw),
            "train_selected_before_augmentation": class_counts(train_records),
            "train_after_three_way_augmentation": len(train_records) * 3,
            "validation": class_counts(validation_records),
            "test_untouched": class_counts(test_records),
        },
        "knot_counts": {
            "train": unique_knot_count(train_raw),
            "validation": unique_knot_count(validation_records),
            "test": unique_knot_count(test_records),
        },
        "model": model.config.to_dict(),
        "parameters": {
            "total": parameter_total,
            "trainable": parameter_trainable,
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "batch_size": args.batch_size,
            "evaluation_batch_size": args.evaluation_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "threshold": args.threshold,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "scheduler": "ReduceLROnPlateau",
            "normalization": "per patch nonzero z score",
            "augmentation": [
                "original",
                "tangential_flip",
                "longitudinal_flip",
            ],
            "class_balance": "random majority downsampling on training only",
            "amp_enabled": amp_enabled,
            "device": str(device),
        },
        "method_provenance": {
            "published": [
                "160 by 80 by 80 full block",
                "11 by 80 by 80 central slice classification input",
                "sound to dead monotonic label propagation",
            ],
            "from_accompanying_thesis": [
                "three convolutional blocks",
                "two convolutions and one pooling layer per block",
                "tangential and longitudinal flip augmentation",
                "class balancing",
            ],
            "reconstructed_because_not_published": [
                "Adam optimizer and binary cross entropy for this classifier",
                "16 32 64 channels",
                "3 by 3 by 3 kernels",
                "ReLU activations",
                "2 by 2 by 2 max pooling",
                "learning rate 0.0002 borrowed from the study segmentation network",
                "batch size 32",
                "per patch nonzero z score normalization",
                "decision threshold 0.5",
            ],
        },
    }
    write_json(
        os.path.join(args.output_dir, "experiment_config.json"),
        experiment_config,
    )

    logger.info("Device: %s | AMP: %s", device, amp_enabled)
    logger.info("Train trees: %s", TRAIN_TREE_NUMBERS)
    logger.info("Validation trees: %s", VALIDATION_TREE_NUMBERS)
    logger.info("Test trees kept untouched: %s", TEST_TREE_NUMBERS)
    logger.info(
        "Train patches: %d raw, %d balanced, %d after augmentation",
        len(train_raw),
        len(train_records),
        len(train_loader.dataset),
    )
    logger.info(
        "Validation patches: %d | Held out test patches: %d",
        len(validation_records),
        len(test_records),
    )
    logger.info("Trainable parameters: %d", parameter_trainable)

    start_epoch = 1
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        validate_resume_checkpoint(checkpoint, args)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        logger.info("Resumed after epoch %d", start_epoch - 1)

    history_path = os.path.join(args.output_dir, "history.csv")
    best_path = os.path.join(args.output_dir, "best_model.pt")
    last_path = os.path.join(args.output_dir, "last_checkpoint.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            amp_enabled,
            args.threshold,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            amp_enabled,
            args.threshold,
        )

        validation_loss = float(validation_metrics["loss"])
        improved = validation_loss < best_validation_loss - 1e-8
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(validation_loss)

        payload = checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_validation_loss=best_validation_loss,
            epochs_without_improvement=epochs_without_improvement,
            args=args,
            validation_metrics=validation_metrics,
        )
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(payload, best_path)

        history_row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_auc": train_metrics["roc_auc"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics[
                "balanced_accuracy"
            ],
            "validation_auc": validation_metrics["roc_auc"],
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        append_history(history_path, history_row)
        logger.info(
            "Epoch %03d | train loss %.5f acc %.4f | val loss %.5f "
            "acc %.4f auc %.4f | lr %.2e | best %s",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            validation_metrics["loss"],
            validation_metrics["accuracy"],
            validation_metrics["roc_auc"],
            optimizer.param_groups[0]["lr"],
            "yes" if improved else "no",
        )

        if epochs_without_improvement >= args.early_stop_patience:
            logger.info(
                "Early stopping after %d epochs without improvement.",
                epochs_without_improvement,
            )
            break

    if not os.path.isfile(best_path):
        raise RuntimeError("Training ended without creating best_model.pt.")
    logger.info("Training complete. Run evaluate_sound_dead.py next.")


if __name__ == "__main__":
    main()
