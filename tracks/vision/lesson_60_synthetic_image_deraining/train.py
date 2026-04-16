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
from dlhub.vision.super_resolution import compute_psnr

from .data import DataConfig, get_dataloaders
from .model import DerainingModel, ModelConfig, build_model, deraining_loss


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
    reconstruction_loss: float
    rain_loss: float
    psnr: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(description="Lesson 60 (Vision): synthetic image deraining.")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--min-shapes", type=int, default=2)
    parser.add_argument("--max-shapes", type=int, default=5)
    parser.add_argument("--rain-lines-min", type=int, default=8)
    parser.add_argument("--rain-lines-max", type=int, default=18)
    parser.add_argument("--rain-length-min", type=int, default=6)
    parser.add_argument("--rain-length-max", type=int, default=12)
    parser.add_argument("--rain-strength-min", type=float, default=0.10)
    parser.add_argument("--rain-strength-max", type=float, default=0.30)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=3)

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
        min_shapes=args.min_shapes,
        max_shapes=args.max_shapes,
        rain_lines_min=args.rain_lines_min,
        rain_lines_max=args.rain_lines_max,
        rain_length_min=args.rain_length_min,
        rain_length_max=args.rain_length_max,
        rain_strength_min=args.rain_strength_min,
        rain_strength_max=args.rain_strength_max,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    return train_cfg, data_cfg, model_cfg


def _run_epoch(
    *,
    model: DerainingModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
) -> Stats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_rain_loss = 0.0
    total_psnr = 0.0
    total_batches = 0

    for batch_idx, (rainy, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        rainy = rainy.to(device)
        target_batch = {key: value.to(device) for key, value in targets.items()}
        outputs = model(rainy)
        loss, parts = deraining_loss(outputs, target_batch)
        psnr = compute_psnr(outputs["restored"].detach(), target_batch["clean"].detach())

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_reconstruction_loss += float(parts["reconstruction_loss"])
        total_rain_loss += float(parts["rain_loss"])
        total_psnr += float(psnr.item())
        total_batches += 1

    if total_batches == 0:
        return Stats(loss=0.0, reconstruction_loss=0.0, rain_loss=0.0, psnr=0.0)
    return Stats(
        loss=total_loss / total_batches,
        reconstruction_loss=total_reconstruction_loss / total_batches,
        rain_loss=total_rain_loss / total_batches,
        psnr=total_psnr / total_batches,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_60_synthetic_image_deraining",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.image_deraining", log_file=paths.logs_dir / "train.log")
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
            "Epoch %d/%d | train loss %.6f | train psnr %.2f | eval loss %.6f | eval psnr %.2f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.psnr,
            eval_stats.loss,
            eval_stats.psnr,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_reconstruction_loss": train_stats.reconstruction_loss,
                "train_rain_loss": train_stats.rain_loss,
                "train_psnr": train_stats.psnr,
                "eval_loss": eval_stats.loss,
                "eval_reconstruction_loss": eval_stats.reconstruction_loss,
                "eval_rain_loss": eval_stats.rain_loss,
                "eval_psnr": eval_stats.psnr,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_60_synthetic_image_deraining"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_60_synthetic_image_deraining.train"
        )

    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
