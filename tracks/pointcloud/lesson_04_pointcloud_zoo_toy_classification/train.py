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
from dlhub.training.loop import evaluate_classifier, fit_classifier

from .data import DataConfig, get_dataloaders
from .model import ModelConfig, build_model, list_supported_arches


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"

    arch: str = "pointnet"
    width_mult: float = 1.0
    dropout: float = 0.1


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(description="Lesson 04 (PointCloud): local zoo backbones on toy clouds.")

    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--arch", type=str, default="pointnet", help="Any arch from dlhub.pointcloud.local_zoo (try --list-arch).")
    parser.add_argument("--list-arch", action="store_true", help="Print supported architectures and exit.")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    args = parser.parse_args()

    if args.list_arch:
        print("\n".join(list_supported_arches()))
        raise SystemExit(0)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        arch=args.arch,
        width_mult=args.width_mult,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        num_points=args.num_points,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
    )
    return train_cfg, data_cfg


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(track="pointcloud", lesson="lesson_04_pointcloud_zoo_toy_classification", run_name=train_cfg.run_name)
    logger = get_logger("pointcloud.zoo", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Arch: %s", train_cfg.arch)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = build_model(
        ModelConfig(
            arch=str(train_cfg.arch),
            in_channels=3,
            num_classes=2,
            num_points=int(data_cfg.num_points),
            width_mult=float(train_cfg.width_mult),
            dropout=float(train_cfg.dropout),
        )
    ).to(device_info.torch_device)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = fit_classifier(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device_info.torch_device,
            max_batches=train_cfg.max_train_batches,
        )
        eval_stats = evaluate_classifier(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )

        logger.info(
            "Epoch %d/%d | train loss %.4f acc %.3f | eval loss %.4f acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.accuracy,
            eval_stats.loss,
            eval_stats.accuracy,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_acc": train_stats.accuracy,
                "eval_loss": eval_stats.loss,
                "eval_acc": eval_stats.accuracy,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "pointcloud", "lesson": "lesson_04_pointcloud_zoo_toy_classification"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.pointcloud.lesson_04_pointcloud_zoo_toy_classification.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())

