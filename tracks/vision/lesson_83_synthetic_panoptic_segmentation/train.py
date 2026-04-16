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
from .model import (
    ModelConfig,
    PanopticSegmentationModel,
    panoptic_segmentation_loss,
    semantic_pixel_accuracy,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 83 (Vision): synthetic panoptic segmentation."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-thing-classes", type=int, default=3)
    parser.add_argument("--num-stuff-classes", type=int, default=2)
    parser.add_argument("--max-instances", type=int, default=2)

    parser.add_argument("--family", type=str, default="panoptic_fpn")
    parser.add_argument("--variant", type=str, default="panoptic_fpn_tiny")
    parser.add_argument("--width-mult", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
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
        num_thing_classes=args.num_thing_classes,
        num_stuff_classes=args.num_stuff_classes,
        max_instances=args.max_instances,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        num_thing_classes=args.num_thing_classes,
        num_stuff_classes=args.num_stuff_classes,
        max_instances=args.max_instances,
        family=args.family,
        variant=args.variant,
        width_mult=args.width_mult,
    )
    return train_cfg, data_cfg, model_cfg


def _run_epoch(
    *,
    model: PanopticSegmentationModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
    max_instances: int,
) -> tuple[float, float, float, float, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_sem = 0.0
    total_cls = 0.0
    total_mask = 0.0
    total_acc = 0.0
    total_batches = 0

    for step, (images, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break
        images = images.to(device)
        batch_targets = {
            "semantic_labels": targets["semantic_labels"].to(device),
            "instance_masks": targets["instance_masks"].to(device),
            "instance_classes": targets["instance_classes"].to(device),
        }

        outputs = model(images)
        loss, parts = panoptic_segmentation_loss(
            outputs, batch_targets, max_instances=int(max_instances)
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_sem += float(parts["semantic_loss"])
        total_cls += float(parts["instance_cls_loss"])
        total_mask += float(parts["instance_mask_loss"])
        total_acc += semantic_pixel_accuracy(
            outputs["semantic_logits"].detach(), batch_targets["semantic_labels"].detach()
        )
        total_batches += 1

    denom = max(1, total_batches)
    return (
        total_loss / denom,
        total_sem / denom,
        total_cls / denom,
        total_mask / denom,
        total_acc / denom,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_83_synthetic_panoptic_segmentation",
        run_name=train_cfg.run_name,
    )
    logger = get_logger(
        "vision.panoptic_segmentation", log_file=paths.logs_dir / "train.log"
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = PanopticSegmentationModel(model_cfg).to(device_info.torch_device)
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
        train_loss, train_sem, train_cls, train_mask, train_acc = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device_info.torch_device,
            max_batches=train_cfg.max_train_batches,
            max_instances=int(model_cfg.max_instances),
        )
        eval_loss, eval_sem, eval_cls, eval_mask, eval_acc = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
            max_instances=int(model_cfg.max_instances),
        )
        logger.info(
            (
                "Epoch %d/%d | train loss %.4f sem %.4f cls %.4f mask %.4f acc %.3f | "
                "eval loss %.4f sem %.4f cls %.4f mask %.4f acc %.3f"
            ),
            epoch,
            train_cfg.epochs,
            train_loss,
            train_sem,
            train_cls,
            train_mask,
            train_acc,
            eval_loss,
            eval_sem,
            eval_cls,
            eval_mask,
            eval_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_semantic_loss": train_sem,
                "train_instance_cls_loss": train_cls,
                "train_instance_mask_loss": train_mask,
                "train_semantic_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_semantic_loss": eval_sem,
                "eval_instance_cls_loss": eval_cls,
                "eval_instance_mask_loss": eval_mask,
                "eval_semantic_acc": eval_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_83_synthetic_panoptic_segmentation"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_83_synthetic_panoptic_segmentation.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())

