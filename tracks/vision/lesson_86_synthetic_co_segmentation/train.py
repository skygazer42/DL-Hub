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
from .model import CoSegmentationModel, ModelConfig, build_model, co_segmentation_loss, mask_iou


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
    cross_entropy: float
    dice_loss: float
    iou: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 86 (Vision): synthetic co-segmentation from grouped images."
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--set-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--arch", type=str, default="coseg:siamese_coseg_small")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
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
        set_size=args.set_size,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        num_classes=2,
        set_size=args.set_size,
        image_size=args.image_size,
        arch=args.arch,
        width_mult=args.width_mult,
        dropout=args.dropout,
    )
    return train_cfg, data_cfg, model_cfg


def _run_epoch(
    *,
    model: CoSegmentationModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
) -> Stats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images = images.to(device)
        target_batch = {k: v.to(device) for k, v in targets.items()}
        outputs = model(images)
        loss, parts = co_segmentation_loss(outputs, target_batch)
        iou = mask_iou(outputs["mask"].detach(), target_batch["mask"].detach())

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_ce += float(parts["cross_entropy"])
        total_dice += float(parts["dice_loss"])
        total_iou += float(iou)
        total_batches += 1

    if total_batches == 0:
        return Stats(loss=0.0, cross_entropy=0.0, dice_loss=0.0, iou=0.0)
    return Stats(
        loss=total_loss / total_batches,
        cross_entropy=total_ce / total_batches,
        dice_loss=total_dice / total_batches,
        iou=total_iou / total_batches,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(int(train_cfg.seed))
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_86_synthetic_co_segmentation",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.co_segmentation", log_file=paths.logs_dir / "train.log")
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
            "Epoch %d/%d | train loss %.5f iou %.3f | eval loss %.5f iou %.3f",
            epoch,
            int(train_cfg.epochs),
            train_stats.loss,
            train_stats.iou,
            eval_stats.loss,
            eval_stats.iou,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_cross_entropy": train_stats.cross_entropy,
                "train_dice_loss": train_stats.dice_loss,
                "train_iou": train_stats.iou,
                "eval_loss": eval_stats.loss,
                "eval_cross_entropy": eval_stats.cross_entropy,
                "eval_dice_loss": eval_stats.dice_loss,
                "eval_iou": eval_stats.iou,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_86_synthetic_co_segmentation"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_86_synthetic_co_segmentation.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
