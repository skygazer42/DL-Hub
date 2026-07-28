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

from .data import DataConfig, Vocab, get_dataloaders
from .model import (
    AudioGroundedRetrievalConfig,
    CompactAudioGroundedRetrievalModel,
    clip_contrastive_loss,
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
    text_width: int = 32
    fusion_width: int = 48
    init_temperature: float = 0.07


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description=(
            "Lesson 21 (Multimodal): compact audio-grounded retrieval from language queries "
            "to paired audio-video segments."
        )
    )

    parser.add_argument("--num-samples", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--num-mel-bins", type=int, default=24)
    parser.add_argument("--num-audio-steps", type=int, default=12)
    parser.add_argument("--max-text-length", type=int, default=12)
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
    parser.add_argument("--text-width", type=int, default=32)
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
        text_width=args.text_width,
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
        max_text_length=args.max_text_length,
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
    model: CompactAudioGroundedRetrievalModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_examples = 0
    total_loss = 0.0
    total_clip_to_query = 0.0
    total_query_to_clip = 0.0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        moved = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(moved)
            loss = clip_contrastive_loss(outputs["logits_per_clip"], outputs["logits_per_query"])
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(moved["video"].shape[0])
        clip_to_query, query_to_clip = retrieval_accuracy(
            outputs["logits_per_clip"], outputs["logits_per_query"]
        )
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_clip_to_query += clip_to_query * batch_size
        total_query_to_clip += query_to_clip * batch_size

    if total_examples == 0:
        return {"loss": 0.0, "clip_to_query_acc": 0.0, "query_to_clip_acc": 0.0}

    return {
        "loss": total_loss / total_examples,
        "clip_to_query_acc": total_clip_to_query / total_examples,
        "query_to_clip_acc": total_query_to_clip / total_examples,
    }


@torch.no_grad()
def _write_samples(
    *,
    model: CompactAudioGroundedRetrievalModel,
    loader,
    device: torch.device,
    vocab: Vocab,
    out_path,
    epoch: int,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    moved = _move_batch(batch, device)
    outputs = model(moved)
    predicted = outputs["logits_per_clip"].argmax(dim=-1).cpu().tolist()
    row = {
        "epoch": int(epoch),
        "vocab_size": int(vocab.size),
        "queries": list(batch["query_text"][:4]),
        "events": list(batch["event_name"][:4]),
        "motions": list(batch["motion_name"][:4]),
        "segments": [int(s) for s in batch["segment_id"][:4].tolist()],
        "predicted_query_index": predicted[:4],
    }
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="multimodal",
        lesson="lesson_21_audio_grounded_retrieval",
        run_name=train_cfg.run_name,
    )
    logger = get_logger(
        "multimodal.audio_grounded_retrieval_compact",
        log_file=paths.logs_dir / "train.log",
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = CompactAudioGroundedRetrievalModel(
        AudioGroundedRetrievalConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_text_length=int(data_cfg.max_text_length),
            num_frames=int(data_cfg.num_frames),
            image_size=int(data_cfg.image_size),
            num_mel_bins=int(data_cfg.num_mel_bins),
            num_audio_steps=int(data_cfg.num_audio_steps),
            embed_dim=int(train_cfg.embed_dim),
            vision_width=int(train_cfg.vision_width),
            audio_width=int(train_cfg.audio_width),
            text_width=int(train_cfg.text_width),
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
    write_json(paths.run_dir / "vocab.json", vocab.to_dict())

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
            vocab=vocab,
            out_path=samples_path,
            epoch=epoch,
        )
        logger.info(
            "Epoch %d/%d | train loss %.4f c2q %.3f q2c %.3f | eval loss %.4f c2q %.3f q2c %.3f",
            epoch,
            train_cfg.epochs,
            train_stats["loss"],
            train_stats["clip_to_query_acc"],
            train_stats["query_to_clip_acc"],
            eval_stats["loss"],
            eval_stats["clip_to_query_acc"],
            eval_stats["query_to_clip_acc"],
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": int(epoch),
                "train_loss": train_stats["loss"],
                "train_clip_to_query_acc": train_stats["clip_to_query_acc"],
                "train_query_to_clip_acc": train_stats["query_to_clip_acc"],
                "eval_loss": eval_stats["loss"],
                "eval_clip_to_query_acc": eval_stats["clip_to_query_acc"],
                "eval_query_to_clip_acc": eval_stats["query_to_clip_acc"],
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
            "lesson": "lesson_21_audio_grounded_retrieval",
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.multimodal.lesson_21_audio_grounded_retrieval.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
