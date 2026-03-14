from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from dlhub.checkpoint import save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed

from .data import DataConfig, get_dataloaders
from .model import (
    GroundedSamLossConfig,
    GroundedSamModelConfig,
    ToyGroundedSamModel,
    foreground_accuracy,
    grounded_sam_loss,
    mask_dice_score,
    mask_iou,
    presence_accuracy,
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
    dice_weight: float = 1.0


@dataclass(frozen=True)
class Stats:
    loss: float
    presence_loss: float
    mask_bce_loss: float
    mask_dice_loss: float
    presence_acc: float
    mask_iou: float
    dice: float
    foreground_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 11 (Multimodal): Grounded-SAM-lite open-vocabulary segmentation."
    )

    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--mask-size", type=int, default=8)
    parser.add_argument("--max-text-length", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-objects", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=4)

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
    parser.add_argument("--dice-weight", type=float, default=1.0)

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
        dice_weight=args.dice_weight,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mask_size=args.mask_size,
        max_text_length=args.max_text_length,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
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
    model: ToyGroundedSamModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    dice_weight: float,
) -> Stats:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_positive = 0
    total_loss = 0.0
    total_presence_loss = 0.0
    total_mask_bce_loss = 0.0
    total_mask_dice_loss = 0.0
    total_presence_acc = 0.0
    total_mask_iou = 0.0
    total_dice = 0.0
    total_foreground_acc = 0.0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        batch = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if is_train:
            outputs = model(batch)
            losses = grounded_sam_loss(
                presence_logit=outputs["presence_logit"],
                mask_logits=outputs["mask_logits"],
                target_present=batch["target_present"],
                target_mask=batch["target_mask"],
                cfg=GroundedSamLossConfig(dice_weight=dice_weight),
            )
        else:
            with torch.no_grad():
                outputs = model(batch)
                losses = grounded_sam_loss(
                    presence_logit=outputs["presence_logit"],
                    mask_logits=outputs["mask_logits"],
                    target_present=batch["target_present"],
                    target_mask=batch["target_mask"],
                    cfg=GroundedSamLossConfig(dice_weight=dice_weight),
                )

        if is_train:
            losses["loss"].backward()
            optimizer.step()

        batch_size = int(batch["image"].shape[0])
        positive_count = int((batch["target_present"] > 0.5).to(torch.long).sum().item())

        total_examples += batch_size
        total_positive += positive_count
        total_loss += float(losses["loss"].item()) * batch_size
        total_presence_loss += float(losses["presence_loss"].item()) * batch_size
        total_mask_bce_loss += float(losses["mask_bce_loss"].item()) * batch_size
        total_mask_dice_loss += float(losses["mask_dice_loss"].item()) * batch_size
        total_presence_acc += (
            presence_accuracy(outputs["presence_logit"], batch["target_present"]) * batch_size
        )

        if positive_count > 0:
            total_mask_iou += (
                mask_iou(outputs["mask_logits"], batch["target_mask"], batch["target_present"])
                * positive_count
            )
            total_dice += (
                mask_dice_score(
                    outputs["mask_logits"], batch["target_mask"], batch["target_present"]
                )
                * positive_count
            )
            total_foreground_acc += (
                foreground_accuracy(
                    outputs["mask_logits"], batch["target_mask"], batch["target_present"]
                )
                * positive_count
            )

    if total_examples == 0:
        return Stats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    positive_denom = float(total_positive) if total_positive > 0 else 1.0
    return Stats(
        loss=total_loss / total_examples,
        presence_loss=total_presence_loss / total_examples,
        mask_bce_loss=total_mask_bce_loss / total_examples,
        mask_dice_loss=total_mask_dice_loss / total_examples,
        presence_acc=total_presence_acc / total_examples,
        mask_iou=total_mask_iou / positive_denom,
        dice=total_dice / positive_denom,
        foreground_acc=total_foreground_acc / positive_denom,
    )


@torch.no_grad()
def _write_samples(
    *,
    model: ToyGroundedSamModel,
    loader,
    device: torch.device,
    out_path: Path,
    epoch: int,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    moved = _move_batch(batch, device)
    outputs = model(moved)
    pred_present_prob = torch.sigmoid(outputs["presence_logit"]).cpu().tolist()
    pred_present_label = [1 if float(prob) >= 0.5 else 0 for prob in pred_present_prob]
    pred_mask = outputs["pred_mask"].cpu()
    target_mask = batch["target_mask"]
    target_present = batch["target_present"].tolist()

    rows = []
    for idx in range(min(4, int(pred_mask.shape[0]))):
        sample_target_present = float(target_present[idx])
        sample_iou = 0.0
        if sample_target_present > 0.5:
            sample_iou = mask_iou(
                outputs["mask_logits"][idx : idx + 1],
                moved["target_mask"][idx : idx + 1],
                moved["target_present"][idx : idx + 1],
            )
        rows.append(
            {
                "epoch": int(epoch),
                "query": batch["query_text"][idx],
                "target_present": int(round(sample_target_present)),
                "pred_present": int(pred_present_label[idx]),
                "pred_present_prob": round(float(pred_present_prob[idx]), 4),
                "target_foreground_ratio": round(float(target_mask[idx].mean().item()), 4),
                "pred_foreground_ratio": round(float(pred_mask[idx].mean().item()), 4),
                "sample_iou": round(float(sample_iou), 4),
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
        lesson="lesson_11_grounded_sam_toy_open_vocab_segmentation",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("multimodal.grounded_sam_toy", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = ToyGroundedSamModel(
        GroundedSamModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            image_size=int(data_cfg.image_size),
            mask_size=int(data_cfg.mask_size),
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
            dice_weight=float(train_cfg.dice_weight),
        )
        eval_stats = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            max_batches=train_cfg.max_eval_batches,
            dice_weight=float(train_cfg.dice_weight),
        )

        _write_samples(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            out_path=samples_path,
            epoch=epoch,
        )

        logger.info(
            "Epoch %d/%d | train loss %.4f present %.4f mask_bce %.4f mask_dice %.4f pacc %.3f iou %.3f dice %.3f fg %.3f | eval loss %.4f pacc %.3f iou %.3f dice %.3f fg %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.presence_loss,
            train_stats.mask_bce_loss,
            train_stats.mask_dice_loss,
            train_stats.presence_acc,
            train_stats.mask_iou,
            train_stats.dice,
            train_stats.foreground_acc,
            eval_stats.loss,
            eval_stats.presence_acc,
            eval_stats.mask_iou,
            eval_stats.dice,
            eval_stats.foreground_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_presence_loss": train_stats.presence_loss,
                "train_mask_bce_loss": train_stats.mask_bce_loss,
                "train_mask_dice_loss": train_stats.mask_dice_loss,
                "train_presence_acc": train_stats.presence_acc,
                "train_mask_iou": train_stats.mask_iou,
                "train_dice": train_stats.dice,
                "train_foreground_acc": train_stats.foreground_acc,
                "eval_loss": eval_stats.loss,
                "eval_presence_loss": eval_stats.presence_loss,
                "eval_mask_bce_loss": eval_stats.mask_bce_loss,
                "eval_mask_dice_loss": eval_stats.mask_dice_loss,
                "eval_presence_acc": eval_stats.presence_acc,
                "eval_mask_iou": eval_stats.mask_iou,
                "eval_dice": eval_stats.dice,
                "eval_foreground_acc": eval_stats.foreground_acc,
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
            "lesson": "lesson_11_grounded_sam_toy_open_vocab_segmentation",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.multimodal.lesson_11_grounded_sam_toy_open_vocab_segmentation.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
