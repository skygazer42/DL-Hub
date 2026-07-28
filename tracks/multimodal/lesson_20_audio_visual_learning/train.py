from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import torch

from dlhub.checkpoint import save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed

from .data import DataConfig, get_dataloaders, num_events, num_motions
from .model import (
    AudioVisualLearningConfig,
    CompactAudioVisualLearningModel,
    classification_accuracy,
    multitask_loss,
    retrieval_accuracy,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    embed_dim: int = 32
    vision_width: int = 32
    audio_width: int = 32
    fusion_width: int = 48
    init_temperature: float = 0.07


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 20 (Multimodal): compact audio-visual learning with contrastive alignment and fused event prediction."
    )

    parser.add_argument("--num-samples", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--num-mel-bins", type=int, default=24)
    parser.add_argument("--num-audio-steps", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--vision-width", type=int, default=32)
    parser.add_argument("--audio-width", type=int, default=32)
    parser.add_argument("--fusion-width", type=int, default=48)
    parser.add_argument("--init-temperature", type=float, default=0.07)

    args = parser.parse_args()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        embed_dim=args.embed_dim,
        vision_width=args.vision_width,
        audio_width=args.audio_width,
        fusion_width=args.fusion_width,
        init_temperature=args.init_temperature,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        image_size=args.image_size,
        num_mel_bins=args.num_mel_bins,
        num_audio_steps=args.num_audio_steps,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
    )
    return train_cfg, data_cfg


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _run_epoch(
    *,
    model: CompactAudioVisualLearningModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_examples = 0
    total_loss = 0.0
    total_v2a = 0.0
    total_a2v = 0.0
    total_event_acc = 0.0
    total_motion_acc = 0.0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        moved = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(moved)
            loss = multitask_loss(outputs, moved)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(moved["video"].shape[0])
        v2a, a2v = retrieval_accuracy(outputs["logits_per_video"], outputs["logits_per_audio"])
        event_acc, motion_acc = classification_accuracy(
            outputs["event_logits"],
            outputs["motion_logits"],
            moved["event_id"],
            moved["motion_id"],
        )
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_v2a += v2a * batch_size
        total_a2v += a2v * batch_size
        total_event_acc += event_acc * batch_size
        total_motion_acc += motion_acc * batch_size

    if total_examples == 0:
        return {
            "loss": 0.0,
            "video_to_audio_acc": 0.0,
            "audio_to_video_acc": 0.0,
            "event_acc": 0.0,
            "motion_acc": 0.0,
        }

    return {
        "loss": total_loss / total_examples,
        "video_to_audio_acc": total_v2a / total_examples,
        "audio_to_video_acc": total_a2v / total_examples,
        "event_acc": total_event_acc / total_examples,
        "motion_acc": total_motion_acc / total_examples,
    }


@torch.no_grad()
def _write_samples(
    *,
    model: CompactAudioVisualLearningModel,
    loader,
    device: torch.device,
    out_path,
    epoch: int,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    moved = _move_batch(batch, device)
    outputs = model(moved)
    predicted_events = outputs["event_logits"].argmax(dim=-1).cpu().tolist()
    predicted_motions = outputs["motion_logits"].argmax(dim=-1).cpu().tolist()
    rows = []
    for idx in range(min(4, len(predicted_events))):
        rows.append(
            {
                "epoch": int(epoch),
                "event_name": batch["event_name"][idx],
                "motion_name": batch["motion_name"][idx],
                "audio_pattern": batch["audio_pattern"][idx],
                "predicted_event_id": int(predicted_events[idx]),
                "predicted_motion_id": int(predicted_motions[idx]),
            }
        )

    with open(out_path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="multimodal",
        lesson="lesson_20_audio_visual_learning",
        run_name=train_cfg.run_name,
    )
    logger = get_logger(
        "multimodal.audio_visual_learning_compact",
        log_file=paths.logs_dir / "train.log",
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = CompactAudioVisualLearningModel(
        AudioVisualLearningConfig(
            num_frames=int(data_cfg.num_frames),
            image_size=int(data_cfg.image_size),
            num_mel_bins=int(data_cfg.num_mel_bins),
            num_audio_steps=int(data_cfg.num_audio_steps),
            num_events=num_events(),
            num_motions=num_motions(),
            embed_dim=int(train_cfg.embed_dim),
            vision_width=int(train_cfg.vision_width),
            audio_width=int(train_cfg.audio_width),
            fusion_width=int(train_cfg.fusion_width),
            init_temperature=float(train_cfg.init_temperature),
        )
    ).to(device_info.torch_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    metrics_path = paths.run_dir / "metrics.jsonl"
    samples_path = paths.run_dir / "samples.jsonl"
    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            max_batches=train_cfg.max_train_batches,
        )
        eval_stats = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            max_batches=train_cfg.max_eval_batches,
        )
        _write_samples(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            out_path=samples_path,
            epoch=epoch,
        )
        logger.info(
            "Epoch %d/%d | train loss %.4f v2a %.3f a2v %.3f event %.3f motion %.3f | eval loss %.4f v2a %.3f a2v %.3f event %.3f motion %.3f",
            epoch,
            train_cfg.epochs,
            train_stats["loss"],
            train_stats["video_to_audio_acc"],
            train_stats["audio_to_video_acc"],
            train_stats["event_acc"],
            train_stats["motion_acc"],
            eval_stats["loss"],
            eval_stats["video_to_audio_acc"],
            eval_stats["audio_to_video_acc"],
            eval_stats["event_acc"],
            eval_stats["motion_acc"],
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": int(epoch),
                "train_loss": train_stats["loss"],
                "train_video_to_audio_acc": train_stats["video_to_audio_acc"],
                "train_audio_to_video_acc": train_stats["audio_to_video_acc"],
                "train_event_acc": train_stats["event_acc"],
                "train_motion_acc": train_stats["motion_acc"],
                "eval_loss": eval_stats["loss"],
                "eval_video_to_audio_acc": eval_stats["video_to_audio_acc"],
                "eval_audio_to_video_acc": eval_stats["audio_to_video_acc"],
                "eval_event_acc": eval_stats["event_acc"],
                "eval_motion_acc": eval_stats["motion_acc"],
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={
            "track": "multimodal",
            "lesson": "lesson_20_audio_visual_learning",
            "num_events": num_events(),
            "num_motions": num_motions(),
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.multimodal.lesson_20_audio_visual_learning.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
