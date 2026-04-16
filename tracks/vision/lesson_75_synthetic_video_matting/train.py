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
from .model import ModelConfig, VideoMattingModel, video_matting_loss, video_matting_mae


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
class TrainStats:
    loss: float
    bce_loss: float
    l1_loss: float
    mae: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 75 (Vision): synthetic video matting with alpha prediction."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--noise-std", type=float, default=0.02)

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
        seq_len=args.seq_len,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        in_channels=args.in_channels,
        noise_std=args.noise_std,
    )
    model_cfg = ModelConfig(
        in_channels=2,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    return train_cfg, data_cfg, model_cfg


def _run_epoch(
    *,
    model: VideoMattingModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
) -> TrainStats:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_bce = 0.0
    total_l1 = 0.0
    total_mae = 0.0
    total_batches = 0

    for step, (videos, trimaps, alphas) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        videos = videos.to(device)
        trimaps = trimaps.to(device)
        alphas = alphas.to(device)
        logits = model(videos, trimaps)
        loss, parts = video_matting_loss(logits, alphas)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(parts["bce_loss"])
        total_l1 += float(parts["l1_loss"])
        total_mae += video_matting_mae(logits.detach(), alphas.detach())
        total_batches += 1

    if total_batches == 0:
        return TrainStats(loss=0.0, bce_loss=0.0, l1_loss=0.0, mae=0.0)
    return TrainStats(
        loss=total_loss / total_batches,
        bce_loss=total_bce / total_batches,
        l1_loss=total_l1 / total_batches,
        mae=total_mae / total_batches,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_75_synthetic_video_matting",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.video_matting", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = VideoMattingModel(model_cfg).to(device_info.torch_device)
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
            "Epoch %d/%d | train loss %.4f mae %.4f | eval loss %.4f mae %.4f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.mae,
            eval_stats.loss,
            eval_stats.mae,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_bce_loss": train_stats.bce_loss,
                "train_l1_loss": train_stats.l1_loss,
                "train_mae": train_stats.mae,
                "eval_loss": eval_stats.loss,
                "eval_bce_loss": eval_stats.bce_loss,
                "eval_l1_loss": eval_stats.l1_loss,
                "eval_mae": eval_stats.mae,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_75_synthetic_video_matting"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_75_synthetic_video_matting.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())

