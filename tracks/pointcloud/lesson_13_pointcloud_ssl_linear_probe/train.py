from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from dlhub.checkpoint import load_checkpoint, save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed
from dlhub.training.loop import evaluate_classifier, fit_classifier

from .data import DataConfig, get_dataloaders
from .model import ModelConfig, build_model, list_supported_ssl_arches


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 2e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"

    ssl_arch: str = "simclr_pointnet:simclr_pointnet_small"
    ssl_checkpoint: str = ""
    ssl_dropout: float = 0.0
    freeze_ssl: bool = True


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(description="Lesson 13 (PointCloud): linear probe on SSL encoders (toy-first).")

    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument(
        "--ssl-arch",
        type=str,
        default="simclr_pointnet:simclr_pointnet_small",
        help="Supported: simclr_pointnet:<variant> | byol_pointnet:<variant> | vicreg_pointnet:<variant>",
    )
    parser.add_argument("--ssl-checkpoint", type=str, default="", help="Path to a checkpoint.pt from lesson 09/11/12.")
    parser.add_argument("--ssl-dropout", type=float, default=0.0)
    parser.add_argument("--freeze-ssl", type=int, default=1, help="1=freeze SSL backbone (linear probe), 0=fine-tune.")
    parser.add_argument("--list-ssl-arch", action="store_true", help="Print supported SSL arch IDs and exit.")

    args = parser.parse_args()

    if args.list_ssl_arch:
        print("\n".join(list_supported_ssl_arches()))
        raise SystemExit(0)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        ssl_arch=args.ssl_arch,
        ssl_checkpoint=args.ssl_checkpoint,
        ssl_dropout=args.ssl_dropout,
        freeze_ssl=bool(int(args.freeze_ssl)),
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        num_points=args.num_points,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(track="pointcloud", lesson="lesson_13_pointcloud_ssl_linear_probe", run_name=train_cfg.run_name)
    logger = get_logger("pointcloud.ssl_linear_probe", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("SSL arch: %s", train_cfg.ssl_arch)
    if train_cfg.ssl_checkpoint:
        logger.info("SSL checkpoint: %s", train_cfg.ssl_checkpoint)
    logger.info("Freeze SSL: %s", train_cfg.freeze_ssl)
    logger.info("Outputs: %s", paths.run_dir)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    train_loader, val_loader = get_dataloaders(data_cfg)

    model_cfg = ModelConfig(
        ssl_arch=str(train_cfg.ssl_arch),
        ssl_dropout=float(train_cfg.ssl_dropout),
        in_channels=3,
        num_classes=2,
        freeze_ssl=bool(train_cfg.freeze_ssl),
    )
    model = build_model(model_cfg).to(device_info.torch_device)

    if train_cfg.ssl_checkpoint:
        ckpt_path = Path(train_cfg.ssl_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--ssl-checkpoint not found: {str(ckpt_path)}")
        load_checkpoint(ckpt_path, model=model.ssl, optimizer=None, map_location=device_info.torch_device)
        logger.info("Loaded SSL weights from %s", ckpt_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )

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
        extra={"track": "pointcloud", "lesson": "lesson_13_pointcloud_ssl_linear_probe"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.pointcloud.lesson_13_pointcloud_ssl_linear_probe.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())

