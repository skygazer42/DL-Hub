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
    BmnModelConfig,
    CompactBmnTemporalGroundingModel,
    end_accuracy,
    recall_at_iou,
    start_accuracy,
    temporal_grounding_loss,
    temporal_iou_metric,
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
    hidden_dim: int = 64
    vision_width: int = 32
    text_dim: int = 32


@dataclass(frozen=True)
class Stats:
    loss: float
    start_loss: float
    end_loss: float
    proposal_loss: float
    start_acc: float
    end_acc: float
    mean_tiou: float
    r1_iou50: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 14 (Multimodal): BMN-lite compact temporal grounding."
    )

    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--max-text-length", type=int, default=16)
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

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--vision-width", type=int, default=32)
    parser.add_argument("--text-dim", type=int, default=32)

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
        hidden_dim=args.hidden_dim,
        vision_width=args.vision_width,
        text_dim=args.text_dim,
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
    model: CompactBmnTemporalGroundingModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> Stats:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss = 0.0
    total_start_loss = 0.0
    total_end_loss = 0.0
    total_proposal_loss = 0.0
    total_start_acc = 0.0
    total_end_acc = 0.0
    total_tiou = 0.0
    total_r1 = 0.0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        batch = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if is_train:
            outputs = model(batch)
            losses = temporal_grounding_loss(
                start_logits=outputs["start_logits"],
                end_logits=outputs["end_logits"],
                proposal_scores=outputs["proposal_scores"],
                start_labels=batch["start_labels"],
                end_labels=batch["end_labels"],
                proposal_labels=batch["proposal_labels"],
                proposal_mask=batch["proposal_mask"],
            )
        else:
            with torch.no_grad():
                outputs = model(batch)
                losses = temporal_grounding_loss(
                    start_logits=outputs["start_logits"],
                    end_logits=outputs["end_logits"],
                    proposal_scores=outputs["proposal_scores"],
                    start_labels=batch["start_labels"],
                    end_labels=batch["end_labels"],
                    proposal_labels=batch["proposal_labels"],
                    proposal_mask=batch["proposal_mask"],
                )

        if is_train:
            losses["loss"].backward()
            optimizer.step()

        batch_size = int(batch["video"].shape[0])
        total_examples += batch_size
        total_loss += float(losses["loss"].item()) * batch_size
        total_start_loss += float(losses["start_loss"].item()) * batch_size
        total_end_loss += float(losses["end_loss"].item()) * batch_size
        total_proposal_loss += float(losses["proposal_loss"].item()) * batch_size
        total_start_acc += start_accuracy(outputs["start_logits"], batch["segment"]) * batch_size
        total_end_acc += end_accuracy(outputs["end_logits"], batch["segment"]) * batch_size
        total_tiou += temporal_iou_metric(outputs["pred_segments"], batch["segment"]) * batch_size
        total_r1 += recall_at_iou(outputs["pred_segments"], batch["segment"], threshold=0.5) * batch_size

    if total_examples == 0:
        return Stats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return Stats(
        loss=total_loss / total_examples,
        start_loss=total_start_loss / total_examples,
        end_loss=total_end_loss / total_examples,
        proposal_loss=total_proposal_loss / total_examples,
        start_acc=total_start_acc / total_examples,
        end_acc=total_end_acc / total_examples,
        mean_tiou=total_tiou / total_examples,
        r1_iou50=total_r1 / total_examples,
    )


@torch.no_grad()
def _write_samples(
    *,
    model: CompactBmnTemporalGroundingModel,
    loader,
    vocab: Vocab,
    device: torch.device,
    out_path,
    epoch: int,
) -> None:
    del vocab
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    moved = _move_batch(batch, device)
    outputs = model(moved)
    pred_segments = outputs["pred_segments"].cpu()
    target_segments = batch["segment"].cpu()

    rows = []
    for idx in range(min(4, int(pred_segments.shape[0]))):
        pred_seg = pred_segments[idx]
        target_seg = target_segments[idx]
        inter = max(0, min(int(pred_seg[1].item()), int(target_seg[1].item())) - max(int(pred_seg[0].item()), int(target_seg[0].item())) + 1)
        union = (int(pred_seg[1].item()) - int(pred_seg[0].item()) + 1) + (int(target_seg[1].item()) - int(target_seg[0].item()) + 1) - inter
        tiou = float(inter) / float(max(1, union))
        rows.append(
            {
                "epoch": int(epoch),
                "query": batch["query_text"][idx],
                "event_type": batch["event_type"][idx],
                "target_segment": [int(v) for v in target_seg.tolist()],
                "pred_segment": [int(v) for v in pred_seg.tolist()],
                "tiou": round(float(tiou), 4),
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
        lesson="lesson_14_bmn_compact_temporal_grounding",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("multimodal.bmn_compact", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = CompactBmnTemporalGroundingModel(
        BmnModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_frames=int(data_cfg.num_frames),
            hidden_dim=int(train_cfg.hidden_dim),
            vision_width=int(train_cfg.vision_width),
            text_dim=int(train_cfg.text_dim),
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
            vocab=vocab,
            device=device_info.torch_device,
            out_path=samples_path,
            epoch=epoch,
        )

        logger.info(
            "Epoch %d/%d | train loss %.4f start %.4f end %.4f prop %.4f sacc %.3f eacc %.3f tiou %.3f r1@0.5 %.3f | eval loss %.4f sacc %.3f eacc %.3f tiou %.3f r1@0.5 %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.start_loss,
            train_stats.end_loss,
            train_stats.proposal_loss,
            train_stats.start_acc,
            train_stats.end_acc,
            train_stats.mean_tiou,
            train_stats.r1_iou50,
            eval_stats.loss,
            eval_stats.start_acc,
            eval_stats.end_acc,
            eval_stats.mean_tiou,
            eval_stats.r1_iou50,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_start_loss": train_stats.start_loss,
                "train_end_loss": train_stats.end_loss,
                "train_proposal_loss": train_stats.proposal_loss,
                "train_start_acc": train_stats.start_acc,
                "train_end_acc": train_stats.end_acc,
                "train_mean_tiou": train_stats.mean_tiou,
                "train_r1_iou50": train_stats.r1_iou50,
                "eval_loss": eval_stats.loss,
                "eval_start_acc": eval_stats.start_acc,
                "eval_end_acc": eval_stats.end_acc,
                "eval_mean_tiou": eval_stats.mean_tiou,
                "eval_r1_iou50": eval_stats.r1_iou50,
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
            "lesson": "lesson_14_bmn_compact_temporal_grounding",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.multimodal.lesson_14_bmn_compact_temporal_grounding.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
