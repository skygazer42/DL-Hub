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
from .model import ModelConfig, ToyTextToVideoModel, text_to_video_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 49 (Generative): toy text-to-video generation."
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)

    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--family", type=str, default="diffusion_t2v")
    parser.add_argument("--variant", type=str, default="diffusion_t2v_tiny")
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
        batch_size=args.batch_size,
        frames=args.frames,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=args.data_seed,
        val_fraction=args.val_fraction,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        family=args.family,
        variant=args.variant,
        width_mult=args.width_mult,
    )
    return train_cfg, data_cfg, model_cfg


def _evaluate(
    model: ToyTextToVideoModel,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_video_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for batch_idx, (prompts, target_video) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            target_video = target_video.to(device)
            outputs = model(prompts)
            loss, parts = text_to_video_loss(outputs, target_video)
            bsz = int(target_video.size(0))
            total_loss += float(loss.item()) * bsz
            total_video_loss += float(parts["video_loss"].item()) * bsz
            total_seen += bsz
    denom = max(1, total_seen)
    return total_loss / denom, total_video_loss / denom


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="generative",
        lesson="lesson_49_toy_text_to_video",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("generative.toy_text_to_video", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "model": dataclass_to_dict(model_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )
    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = ToyTextToVideoModel(model_cfg).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, int(train_cfg.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_video_loss = 0.0
        total_seen = 0

        for batch_idx, (prompts, target_video) in enumerate(train_loader):
            if train_cfg.max_train_batches is not None and batch_idx >= train_cfg.max_train_batches:
                break
            target_video = target_video.to(device_info.torch_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(prompts)
            loss, parts = text_to_video_loss(outputs, target_video)
            loss.backward()
            optimizer.step()
            bsz = int(target_video.size(0))
            total_loss += float(loss.item()) * bsz
            total_video_loss += float(parts["video_loss"].item()) * bsz
            total_seen += bsz

        denom = max(1, total_seen)
        train_loss = total_loss / denom
        train_video_loss = total_video_loss / denom
        eval_loss, eval_video_loss = _evaluate(
            model,
            val_loader,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )

        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_video_loss": train_video_loss,
                "eval_loss": eval_loss,
                "eval_video_loss": eval_video_loss,
            },
        )
        logger.info(
            "Epoch %d/%d | train_loss %.4f | train_video_loss %.4f | eval_loss %.4f",
            epoch,
            train_cfg.epochs,
            train_loss,
            train_video_loss,
            eval_loss,
        )

    sample_prompts, sample_target = next(iter(val_loader))
    sample_prompts = sample_prompts[:8]
    sample_target = sample_target[:8]
    sample_outputs = model(sample_prompts)
    torch.save(
        {
            "prompts": sample_prompts,
            "target_video": sample_target,
            "generated_video": sample_outputs["video"].detach().cpu(),
            "motion": sample_outputs["motion"].detach().cpu(),
        },
        paths.run_dir / "samples.pt",
    )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "generative", "lesson": "lesson_49_toy_text_to_video"},
    )
    return 0


def main() -> int:
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
