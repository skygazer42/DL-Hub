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
from .model import (
    CompactVisionLanguageNavigationModel,
    VisionLanguageNavigationConfig,
    navigation_accuracy,
    navigation_loss,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    hidden_dim: int = 64
    text_dim: int = 32
    vision_width: int = 32


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description=(
            "Lesson 25 (Multimodal): compact vision-language navigation policy "
            "on a compact grid-world task."
        )
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--max-instruction-length", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--text-dim", type=int, default=32)
    parser.add_argument("--vision-width", type=int, default=32)

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
        text_dim=args.text_dim,
        vision_width=args.vision_width,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        max_instruction_length=args.max_instruction_length,
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
    model: CompactVisionLanguageNavigationModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_examples = 0
    total_loss = 0.0
    total_acc = 0.0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        moved = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(moved)
        loss = navigation_loss(outputs["logits"], moved["actions"])

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(moved["actions"].shape[0])
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_acc += navigation_accuracy(outputs["logits"], moved["actions"]) * batch_size

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total_examples, "accuracy": total_acc / total_examples}


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="multimodal",
        lesson="lesson_25_vision_language_navigation",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("multimodal.vision_language_navigation_compact", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = CompactVisionLanguageNavigationModel(
        VisionLanguageNavigationConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_actions=4,
            hidden_dim=int(train_cfg.hidden_dim),
            text_dim=int(train_cfg.text_dim),
            vision_width=int(train_cfg.vision_width),
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
        logger.info(
            "Epoch %d/%d | train loss %.4f acc %.3f | eval loss %.4f acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats["loss"],
            train_stats["accuracy"],
            eval_stats["loss"],
            eval_stats["accuracy"],
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": int(epoch),
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "eval_loss": eval_stats["loss"],
                "eval_accuracy": eval_stats["accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "multimodal", "lesson": "lesson_25_vision_language_navigation"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.multimodal.lesson_25_vision_language_navigation.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
