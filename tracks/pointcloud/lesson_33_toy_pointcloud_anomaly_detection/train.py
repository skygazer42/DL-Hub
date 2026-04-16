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
    anomaly_accuracy,
    anomaly_loss,
    build_model,
    list_supported_arches,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    arch: str = "recon_anomaly3d:recon_anomaly3d_small"
    width_mult: float = 1.0


@dataclass(frozen=True)
class Stats:
    loss: float
    anomaly_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 33 (PointCloud): toy pointcloud anomaly detection."
    )
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--num-points", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--anomaly-fraction", type=float, default=0.35)
    parser.add_argument("--anomaly-scale", type=float, default=0.55)
    parser.add_argument("--jitter-std", type=float, default=0.01)
    parser.add_argument("--p-sphere", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument(
        "--arch",
        type=str,
        default="recon_anomaly3d:recon_anomaly3d_small",
        help="Supported: recon_anomaly3d:<variant> (try --list-arch).",
    )
    parser.add_argument("--list-arch", action="store_true")
    parser.add_argument("--width-mult", type=float, default=1.0)
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
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        num_points=args.num_points,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        anomaly_fraction=args.anomaly_fraction,
        anomaly_scale=args.anomaly_scale,
        jitter_std=args.jitter_std,
        p_sphere=args.p_sphere,
    )
    model_cfg = ModelConfig(
        in_channels=3,
        arch=args.arch,
        variant="",
        width_mult=args.width_mult,
    )
    return train_cfg, data_cfg, model_cfg


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
) -> Stats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch_idx, (points, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        points = points.to(device)
        target_batch = {key: value.to(device) for key, value in targets.items()}
        outputs = model(points)
        loss, _ = anomaly_loss(outputs, target_batch)
        acc = anomaly_accuracy(outputs["global_score"].detach(), target_batch["label"].detach())

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += float(acc)
        total_batches += 1

    if total_batches == 0:
        return Stats(loss=0.0, anomaly_acc=0.0)
    return Stats(loss=total_loss / total_batches, anomaly_acc=total_acc / total_batches)


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="pointcloud",
        lesson="lesson_33_toy_pointcloud_anomaly_detection",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("pointcloud.anomaly", log_file=paths.logs_dir / "train.log")
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
            "Epoch %d/%d | train loss %.6f anomaly_acc %.3f | eval loss %.6f anomaly_acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.anomaly_acc,
            eval_stats.loss,
            eval_stats.anomaly_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_anomaly_acc": train_stats.anomaly_acc,
                "eval_loss": eval_stats.loss,
                "eval_anomaly_acc": eval_stats.anomaly_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "pointcloud", "lesson": "lesson_33_toy_pointcloud_anomaly_detection"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.pointcloud.lesson_33_toy_pointcloud_anomaly_detection.train"
        )

    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
