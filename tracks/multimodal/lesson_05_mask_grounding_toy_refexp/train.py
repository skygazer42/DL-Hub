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
    MaskGroundingLossConfig,
    MaskGroundingModelConfig,
    ToyMaskGroundingModel,
    foreground_accuracy,
    mask_dice_score,
    mask_grounding_loss,
    mask_iou,
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
    bce_loss: float
    dice_loss: float
    mask_iou: float
    dice: float
    foreground_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 05 (Multimodal): mask-grounding-lite referring expressions."
    )

    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--mask-size", type=int, default=8)
    parser.add_argument("--max-text-length", type=int, default=8)
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
    model: ToyMaskGroundingModel,
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
    total_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    total_iou = 0.0
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
            losses = mask_grounding_loss(
                mask_logits=outputs["mask_logits"],
                target_mask=batch["target_mask"],
                cfg=MaskGroundingLossConfig(dice_weight=dice_weight),
            )
        else:
            with torch.no_grad():
                outputs = model(batch)
                losses = mask_grounding_loss(
                    mask_logits=outputs["mask_logits"],
                    target_mask=batch["target_mask"],
                    cfg=MaskGroundingLossConfig(dice_weight=dice_weight),
                )

        if is_train:
            losses["loss"].backward()
            optimizer.step()

        batch_size = int(batch["image"].shape[0])
        total_examples += batch_size
        total_loss += float(losses["loss"].item()) * batch_size
        total_bce_loss += float(losses["bce_loss"].item()) * batch_size
        total_dice_loss += float(losses["dice_loss"].item()) * batch_size
        total_iou += mask_iou(outputs["mask_logits"], batch["target_mask"]) * batch_size
        total_dice += mask_dice_score(outputs["mask_logits"], batch["target_mask"]) * batch_size
        total_foreground_acc += (
            foreground_accuracy(outputs["mask_logits"], batch["target_mask"]) * batch_size
        )

    if total_examples == 0:
        return Stats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return Stats(
        loss=total_loss / total_examples,
        bce_loss=total_bce_loss / total_examples,
        dice_loss=total_dice_loss / total_examples,
        mask_iou=total_iou / total_examples,
        dice=total_dice / total_examples,
        foreground_acc=total_foreground_acc / total_examples,
    )


@torch.no_grad()
def _write_samples(
    *,
    model: ToyMaskGroundingModel,
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
    pred_mask = outputs["pred_mask"].cpu()
    target_mask = batch["target_mask"]

    rows = []
    for idx in range(min(4, int(pred_mask.shape[0]))):
        rows.append(
            {
                "epoch": int(epoch),
                "query": batch["query_text"][idx],
                "target_foreground_ratio": round(float(target_mask[idx].mean().item()), 4),
                "pred_foreground_ratio": round(float(pred_mask[idx].mean().item()), 4),
                "sample_iou": round(
                    mask_iou(
                        outputs["mask_logits"][idx : idx + 1],
                        moved["target_mask"][idx : idx + 1],
                    ),
                    4,
                ),
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
        lesson="lesson_05_mask_grounding_toy_refexp",
        run_name=train_cfg.run_name,
    )
    logger = get_logger(
        "multimodal.mask_grounding_toy",
        log_file=paths.logs_dir / "train.log",
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = ToyMaskGroundingModel(
        MaskGroundingModelConfig(
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
            vocab=vocab,
            device=device_info.torch_device,
            out_path=samples_path,
            epoch=epoch,
        )

        logger.info(
            "Epoch %d/%d | train loss %.4f bce %.4f dice_loss %.4f iou %.3f dice %.3f fg %.3f | eval loss %.4f iou %.3f dice %.3f fg %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.bce_loss,
            train_stats.dice_loss,
            train_stats.mask_iou,
            train_stats.dice,
            train_stats.foreground_acc,
            eval_stats.loss,
            eval_stats.mask_iou,
            eval_stats.dice,
            eval_stats.foreground_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_bce_loss": train_stats.bce_loss,
                "train_dice_loss": train_stats.dice_loss,
                "train_mask_iou": train_stats.mask_iou,
                "train_dice": train_stats.dice,
                "train_foreground_acc": train_stats.foreground_acc,
                "eval_loss": eval_stats.loss,
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
            "lesson": "lesson_05_mask_grounding_toy_refexp",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.multimodal.lesson_05_mask_grounding_toy_refexp.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
