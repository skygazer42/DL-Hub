from __future__ import annotations

import argparse
import math
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
from .model import CrowdCountingRegressor, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"

    hidden_channels: int = 24
    depth: int = 4
    dropout: float = 0.0


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 18 (Vision): synthetic crowd counting with density maps."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-people", type=int, default=4)
    parser.add_argument("--max-people", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--point-sigma", type=float, default=1.6)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--depth", type=int, default=4)
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
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        min_people=args.min_people,
        max_people=args.max_people,
        noise_std=args.noise_std,
        point_sigma=args.point_sigma,
    )
    return train_cfg, data_cfg


def compute_count_metrics(pred_density: torch.Tensor, target_density: torch.Tensor) -> dict[str, float]:
    pred_counts = pred_density.sum(dim=(1, 2, 3))
    target_counts = target_density.sum(dim=(1, 2, 3))
    abs_err = (pred_counts - target_counts).abs()
    sq_err = (pred_counts - target_counts).pow(2)
    return {
        "count_mae": float(abs_err.mean().item()),
        "count_rmse": float(torch.sqrt(sq_err.mean()).item()),
        "pred_count_mean": float(pred_counts.mean().item()),
        "target_count_mean": float(target_counts.mean().item()),
    }


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_abs_err = 0.0
    total_sq_err = 0.0
    total_pred_count = 0.0
    total_target_count = 0.0
    total_items = 0
    steps = 0

    for batch_idx, (x, target_density, _target_count) in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break

        x = x.to(device)
        target_density = target_density.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        if is_train:
            pred_density = model(x)
            loss = criterion(pred_density, target_density)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred_density = model(x)
                loss = criterion(pred_density, target_density)

        pred_counts = pred_density.detach().sum(dim=(1, 2, 3))
        target_counts = target_density.detach().sum(dim=(1, 2, 3))
        batch_size = int(x.shape[0])

        total_loss += float(loss.detach().item())
        total_abs_err += float((pred_counts - target_counts).abs().sum().item())
        total_sq_err += float((pred_counts - target_counts).pow(2).sum().item())
        total_pred_count += float(pred_counts.sum().item())
        total_target_count += float(target_counts.sum().item())
        total_items += batch_size
        steps += 1

    if steps == 0 or total_items == 0:
        raise RuntimeError("No batches were processed. Check dataset size or max_batches.")

    return {
        "loss": total_loss / steps,
        "count_mae": total_abs_err / total_items,
        "count_rmse": math.sqrt(total_sq_err / total_items),
        "pred_count_mean": total_pred_count / total_items,
        "target_count_mean": total_target_count / total_items,
    }


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_18_synthetic_crowd_counting",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.synthetic_crowd_counting", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
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
    model = CrowdCountingRegressor(
        ModelConfig(
            in_channels=1,
            hidden_channels=train_cfg.hidden_channels,
            depth=train_cfg.depth,
            dropout=train_cfg.dropout,
        )
    ).to(device_info.torch_device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device_info.torch_device,
            max_batches=train_cfg.max_train_batches,
        )
        eval_stats = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info(
            "Epoch %d/%d | train loss %.6f count_mae %.3f | eval loss %.6f count_mae %.3f rmse %.3f",
            epoch,
            train_cfg.epochs,
            train_stats["loss"],
            train_stats["count_mae"],
            eval_stats["loss"],
            eval_stats["count_mae"],
            eval_stats["count_rmse"],
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_count_mae": train_stats["count_mae"],
                "train_count_rmse": train_stats["count_rmse"],
                "eval_loss": eval_stats["loss"],
                "eval_count_mae": eval_stats["count_mae"],
                "eval_count_rmse": eval_stats["count_rmse"],
                "eval_pred_count_mean": eval_stats["pred_count_mean"],
                "eval_target_count_mean": eval_stats["target_count_mean"],
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_18_synthetic_crowd_counting"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_18_synthetic_crowd_counting.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
