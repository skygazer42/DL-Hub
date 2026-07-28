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
    ModelConfig,
    CompactVideoTextRetrievalModel,
    clip_contrastive_loss,
    recall_at_k,
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
    temporal_width: int = 32
    text_width: int = 32
    init_temperature: float = 0.07


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 17 (Multimodal): CLIP-style compact video-text retrieval."
    )

    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=20)
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
    parser.add_argument("--temporal-width", type=int, default=32)
    parser.add_argument("--text-width", type=int, default=32)
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
        temporal_width=args.temporal_width,
        text_width=args.text_width,
        init_temperature=args.init_temperature,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        image_size=args.image_size,
        max_text_length=args.max_text_length,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
    )
    return train_cfg, data_cfg


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _run_epoch(
    *,
    model: CompactVideoTextRetrievalModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss = 0.0
    total_v2t = 0.0
    total_t2v = 0.0
    total_v2t_r3 = 0.0
    total_t2v_r3 = 0.0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        batch = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if is_train:
            outputs = model(batch)
            loss = clip_contrastive_loss(outputs["logits_per_video"], outputs["logits_per_text"])
        else:
            with torch.no_grad():
                outputs = model(batch)
                loss = clip_contrastive_loss(outputs["logits_per_video"], outputs["logits_per_text"])

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(outputs["logits_per_video"].shape[0])
        v2t_acc, t2v_acc = retrieval_accuracy(
            outputs["logits_per_video"], outputs["logits_per_text"]
        )
        v2t_r3, t2v_r3 = recall_at_k(
            outputs["logits_per_video"], outputs["logits_per_text"], k=3
        )
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_v2t += float(v2t_acc) * batch_size
        total_t2v += float(t2v_acc) * batch_size
        total_v2t_r3 += float(v2t_r3) * batch_size
        total_t2v_r3 += float(t2v_r3) * batch_size

    if total_examples == 0:
        return {
            "loss": 0.0,
            "video_to_text_acc": 0.0,
            "text_to_video_acc": 0.0,
            "video_to_text_recall_at_3": 0.0,
            "text_to_video_recall_at_3": 0.0,
        }

    return {
        "loss": total_loss / total_examples,
        "video_to_text_acc": total_v2t / total_examples,
        "text_to_video_acc": total_t2v / total_examples,
        "video_to_text_recall_at_3": total_v2t_r3 / total_examples,
        "text_to_video_recall_at_3": total_t2v_r3 / total_examples,
    }


@torch.no_grad()
def _write_samples(
    *,
    model: CompactVideoTextRetrievalModel,
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
    predicted = outputs["logits_per_video"].argmax(dim=-1).cpu().tolist()
    row = {
        "epoch": int(epoch),
        "vocab_size": int(vocab.size),
        "captions": list(batch["caption_text"][:4]),
        "queries": list(batch["query_text"][:4]),
        "motions": list(batch["motion_type"][:4]),
        "predicted_text_index": predicted[:4],
    }
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="multimodal",
        lesson="lesson_17_video_text_retrieval",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("multimodal.video_text_retrieval_compact", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = CompactVideoTextRetrievalModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_text_length=int(data_cfg.max_text_length),
            num_frames=int(data_cfg.num_frames),
            image_size=int(data_cfg.image_size),
            embed_dim=int(train_cfg.embed_dim),
            vision_width=int(train_cfg.vision_width),
            temporal_width=int(train_cfg.temporal_width),
            text_width=int(train_cfg.text_width),
            init_temperature=float(train_cfg.init_temperature),
        )
    ).to(device_info.torch_device)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )
    write_json(paths.run_dir / "vocab.json", vocab.to_dict())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )

    metrics_path = paths.run_dir / "metrics.jsonl"
    samples_path = paths.run_dir / "samples.jsonl"
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

        payload = {"epoch": epoch}
        payload.update({f"train_{key}": value for key, value in train_stats.items()})
        payload.update({f"eval_{key}": value for key, value in eval_stats.items()})
        append_jsonl(metrics_path, payload)
        logger.info("epoch=%s train=%s eval=%s", epoch, train_stats, eval_stats)

        save_checkpoint(
            paths.checkpoints_dir / "checkpoint.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            extra={"train": dataclass_to_dict(train_cfg), "data": dataclass_to_dict(data_cfg)},
        )

    return 0


def main() -> int:
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
