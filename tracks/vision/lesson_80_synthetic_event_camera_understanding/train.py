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
from .model import EventUnderstandingModel, ModelConfig, build_model, event_understanding_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 4
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


@dataclass(frozen=True)
class Stats:
    loss: float
    polarity_loss: float
    motion_loss: float
    depth_loss: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 80 (Vision): synthetic event camera understanding."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--in-channels", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--family", type=str, default="ev_cnn")
    parser.add_argument("--variant", type=str, default="ev_cnn_tiny")
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
        num_bins=args.num_bins,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        noise_std=args.noise_std,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        family=args.family,
        variant=args.variant,
        width_mult=args.width_mult,
    )
    return train_cfg, data_cfg, model_cfg


def _run_epoch(
    *,
    model: EventUnderstandingModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
) -> Stats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_polarity = 0.0
    total_motion = 0.0
    total_depth = 0.0
    total_batches = 0

    for batch_idx, (events, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        events = events.to(device)
        target_batch = {key: value.to(device) for key, value in targets.items()}
        outputs = model(events)
        loss, parts = event_understanding_loss(outputs, target_batch)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_polarity += float(parts["polarity_loss"])
        total_motion += float(parts["motion_loss"])
        total_depth += float(parts["depth_loss"])
        total_batches += 1

    if total_batches == 0:
        return Stats(loss=0.0, polarity_loss=0.0, motion_loss=0.0, depth_loss=0.0)
    return Stats(
        loss=total_loss / total_batches,
        polarity_loss=total_polarity / total_batches,
        motion_loss=total_motion / total_batches,
        depth_loss=total_depth / total_batches,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_80_synthetic_event_camera_understanding",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.event_camera_understanding", log_file=paths.logs_dir / "train.log")
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
            "Epoch %d/%d | train %.5f | eval %.5f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            eval_stats.loss,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_polarity_loss": train_stats.polarity_loss,
                "train_motion_loss": train_stats.motion_loss,
                "train_depth_loss": train_stats.depth_loss,
                "eval_loss": eval_stats.loss,
                "eval_polarity_loss": eval_stats.polarity_loss,
                "eval_motion_loss": eval_stats.motion_loss,
                "eval_depth_loss": eval_stats.depth_loss,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_80_synthetic_event_camera_understanding"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_80_synthetic_event_camera_understanding.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
