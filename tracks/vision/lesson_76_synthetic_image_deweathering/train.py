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
from .model import ModelConfig, DeweatheringModel, build_model, deweathering_loss, list_supported_arches


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    arch: str = "deweather:deweather_cnn_small"
    width_mult: float = 1.0


@dataclass(frozen=True)
class Stats:
    loss: float
    reconstruction_loss: float
    weather_loss: float
    psnr: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 76 (Vision): synthetic image deweathering."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--min-shapes", type=int, default=2)
    parser.add_argument("--max-shapes", type=int, default=5)
    parser.add_argument("--streak-count-min", type=int, default=5)
    parser.add_argument("--streak-count-max", type=int, default=12)
    parser.add_argument("--weather-strength-min", type=float, default=0.12)
    parser.add_argument("--weather-strength-max", type=float, default=0.28)
    parser.add_argument("--haze-strength-min", type=float, default=0.05)
    parser.add_argument("--haze-strength-max", type=float, default=0.20)
    parser.add_argument("--snow-blob-count", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument(
        "--arch",
        type=str,
        default="deweather:deweather_cnn_small",
        help="Supported: deweather:<variant> (try --list-arch).",
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
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        in_channels=args.in_channels,
        min_shapes=args.min_shapes,
        max_shapes=args.max_shapes,
        streak_count_min=args.streak_count_min,
        streak_count_max=args.streak_count_max,
        weather_strength_min=args.weather_strength_min,
        weather_strength_max=args.weather_strength_max,
        haze_strength_min=args.haze_strength_min,
        haze_strength_max=args.haze_strength_max,
        snow_blob_count=args.snow_blob_count,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        arch=args.arch,
        width_mult=args.width_mult,
    )
    return train_cfg, data_cfg, model_cfg


def _run_epoch(
    *,
    model: DeweatheringModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
) -> Stats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_weather_loss = 0.0
    total_psnr = 0.0
    total_batches = 0

    for batch_idx, (weathered, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        weathered = weathered.to(device)
        target_batch = {key: value.to(device) for key, value in targets.items()}
        outputs = model(weathered)
        loss, parts = deweathering_loss(outputs, target_batch)
        psnr = compute_psnr(outputs["restored"].detach(), target_batch["clean"].detach())

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_reconstruction_loss += float(parts["reconstruction_loss"])
        total_weather_loss += float(parts["weather_loss"])
        total_psnr += float(psnr.item())
        total_batches += 1

    if total_batches == 0:
        return Stats(loss=0.0, reconstruction_loss=0.0, weather_loss=0.0, psnr=0.0)
    return Stats(
        loss=total_loss / total_batches,
        reconstruction_loss=total_reconstruction_loss / total_batches,
        weather_loss=total_weather_loss / total_batches,
        psnr=total_psnr / total_batches,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_76_synthetic_image_deweathering",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.image_deweathering", log_file=paths.logs_dir / "train.log")
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
                "train_weather_loss": train_stats.weather_loss,
                "train_psnr": train_stats.psnr,
                "eval_loss": eval_stats.loss,
                "eval_reconstruction_loss": eval_stats.reconstruction_loss,
                "eval_weather_loss": eval_stats.weather_loss,
                "eval_psnr": eval_stats.psnr,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_76_synthetic_image_deweathering"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_76_synthetic_image_deweathering.train"
        )

    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
