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
from dlhub.training.loop import evaluate_regression, fit_regression

from .data import DataConfig, get_dataloaders
from .model import FaceOcclusionRegressor, ModelConfig, mean_occlusion_abs_error


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    hidden_channels: int = 24
    num_blocks: int = 3
    dropout: float = 0.0


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(description="Lesson 41 (Vision): synthetic face occlusion estimation.")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--max-shift", type=float, default=0.08)
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=3)
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
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        in_channels=args.in_channels,
        max_shift=args.max_shift,
        noise_std=args.noise_std,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )
    return train_cfg, data_cfg, model_cfg


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_41_synthetic_face_occlusion_estimation",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.face_occlusion", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = FaceOcclusionRegressor(model_cfg).to(device_info.torch_device)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "model": dataclass_to_dict(model_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = fit_regression(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device_info.torch_device,
            max_batches=train_cfg.max_train_batches,
        )
        eval_stats = evaluate_regression(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )
        eval_mae = _evaluate_mae(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info(
            "Epoch %d/%d | train mse %.6f | eval mse %.6f | eval mae %.4f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            eval_stats.loss,
            eval_mae,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_mse": train_stats.loss,
                "eval_mse": eval_stats.loss,
                "eval_mae": eval_mae,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_41_synthetic_face_occlusion_estimation"},
    )
    return 0


@torch.no_grad()
def _evaluate_mae(
    *,
    model: FaceOcclusionRegressor,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    total_error = 0.0
    total_batches = 0
    for step, (images, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break
        preds = model(images.to(device))
        total_error += mean_occlusion_abs_error(preds.detach().cpu(), targets)
        total_batches += 1
    return total_error / max(1, total_batches)


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_41_synthetic_face_occlusion_estimation.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
