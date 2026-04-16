from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch

from dlhub.checkpoint import save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed

from .data import DataConfig, get_dataloaders
from .model import ModelConfig, TransparentObjectSegmentationModel, build_model, transparent_segmentation_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


@dataclass(frozen=True)
class Stats:
    loss: float
    mask_bce: float
    alpha_l1: float
    boundary_l1: float
    mask_iou: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 79 (Vision): synthetic transparent object segmentation."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=0.8)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--arch", type=str, default="glassseg_toy")
    parser.add_argument("--variant", type=str, default="glassseg_toy_small")
    parser.add_argument("--width-mult", type=float, default=1.0)
    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        in_channels=args.in_channels,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        arch=args.arch,
        variant=args.variant,
        width_mult=args.width_mult,
    )
    return train_cfg, data_cfg, model_cfg


def _compute_mask_iou(mask_pred: torch.Tensor, mask_target: torch.Tensor) -> float:
    pred = mask_pred >= 0.5
    target = mask_target >= 0.5
    intersection = (pred & target).to(torch.float32).sum()
    union = (pred | target).to(torch.float32).sum()
    return float((intersection / (union + 1e-6)).item())


def _run_epoch(
    *,
    model: TransparentObjectSegmentationModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
) -> Stats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mask_bce = 0.0
    total_alpha_l1 = 0.0
    total_boundary_l1 = 0.0
    total_mask_iou = 0.0
    total_batches = 0

    for batch_idx, (image, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        image = image.to(device)
        target_batch = {k: v.to(device) for k, v in targets.items()}
        outputs = model(image)
        loss, parts = transparent_segmentation_loss(outputs, target_batch)
        iou = _compute_mask_iou(outputs["mask"].detach(), target_batch["mask"].detach())

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_mask_bce += float(parts["mask_bce"])
        total_alpha_l1 += float(parts["alpha_l1"])
        total_boundary_l1 += float(parts["boundary_l1"])
        total_mask_iou += float(iou)
        total_batches += 1

    if total_batches == 0:
        return Stats(loss=0.0, mask_bce=0.0, alpha_l1=0.0, boundary_l1=0.0, mask_iou=0.0)

    return Stats(
        loss=total_loss / total_batches,
        mask_bce=total_mask_bce / total_batches,
        alpha_l1=total_alpha_l1 / total_batches,
        boundary_l1=total_boundary_l1 / total_batches,
        mask_iou=total_mask_iou / total_batches,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_79_synthetic_transparent_object_segmentation",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.transparent_object_segmentation", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = build_model(model_cfg).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "model": dataclass_to_dict(model_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device_info.torch_device,
            max_batches=train_cfg.max_train_batches,
        )
        with torch.no_grad():
            eval_stats = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device_info.torch_device,
                max_batches=train_cfg.max_eval_batches,
            )

        logger.info(
            "Epoch %d/%d | train loss %.6f | train iou %.4f | eval loss %.6f | eval iou %.4f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.mask_iou,
            eval_stats.loss,
            eval_stats.mask_iou,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_mask_bce": train_stats.mask_bce,
                "train_alpha_l1": train_stats.alpha_l1,
                "train_boundary_l1": train_stats.boundary_l1,
                "train_mask_iou": train_stats.mask_iou,
                "eval_loss": eval_stats.loss,
                "eval_mask_bce": eval_stats.mask_bce,
                "eval_alpha_l1": eval_stats.alpha_l1,
                "eval_boundary_l1": eval_stats.boundary_l1,
                "eval_mask_iou": eval_stats.mask_iou,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_79_synthetic_transparent_object_segmentation"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_79_synthetic_transparent_object_segmentation.train"
        )

    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
