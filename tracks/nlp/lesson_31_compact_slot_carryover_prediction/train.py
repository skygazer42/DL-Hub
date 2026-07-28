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
from .model import ModelConfig, SlotCarryoverPredictor, compute_slot_carryover_metrics, slot_carryover_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    embed_dim: int = 64
    dropout: float = 0.1


@dataclass(frozen=True)
class EpochStats:
    loss: float
    slot_acc: float
    joint_carry_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(description="Lesson 31 (NLP): compact slot carryover prediction.")
    parser.add_argument("--num-samples", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _run_epoch(
    *,
    model: SlotCarryoverPredictor,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_slot_acc = 0.0
    total_joint_carry_acc = 0.0
    total_batches = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break
        batch = _move(batch, device)
        outputs = model(batch)
        loss = slot_carryover_loss(
            outputs["cuisine_logits"],
            outputs["area_logits"],
            outputs["party_logits"],
            batch["cuisine_labels"],
            batch["area_labels"],
            batch["party_labels"],
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        metrics = compute_slot_carryover_metrics(
            outputs["cuisine_logits"].detach(),
            outputs["area_logits"].detach(),
            outputs["party_logits"].detach(),
            batch["cuisine_labels"].detach(),
            batch["area_labels"].detach(),
            batch["party_labels"].detach(),
        )
        total_loss += float(loss.item())
        total_slot_acc += metrics["slot_acc"]
        total_joint_carry_acc += metrics["joint_carry_acc"]
        total_batches += 1

    denom = max(1, total_batches)
    return EpochStats(
        loss=total_loss / denom,
        slot_acc=total_slot_acc / denom,
        joint_carry_acc=total_joint_carry_acc / denom,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="nlp",
        lesson="lesson_31_compact_slot_carryover_prediction",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("nlp.compact_slot_carryover_prediction", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = SlotCarryoverPredictor(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=train_cfg.embed_dim,
            dropout=train_cfg.dropout,
        )
    ).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )
    write_json(paths.run_dir / "vocab.json", vocab.to_dict())

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    metrics_path = paths.run_dir / "metrics.jsonl"
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
        logger.info(
            "Epoch %d/%d | train loss %.4f slot %.3f joint %.3f | eval loss %.4f slot %.3f joint %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.slot_acc,
            train_stats.joint_carry_acc,
            eval_stats.loss,
            eval_stats.slot_acc,
            eval_stats.joint_carry_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_slot_acc": train_stats.slot_acc,
                "train_joint_carry_acc": train_stats.joint_carry_acc,
                "eval_loss": eval_stats.loss,
                "eval_slot_acc": eval_stats.slot_acc,
                "eval_joint_carry_acc": eval_stats.joint_carry_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "nlp", "lesson": "lesson_31_compact_slot_carryover_prediction", "vocab_size": vocab.size},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.nlp.lesson_31_compact_slot_carryover_prediction.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
