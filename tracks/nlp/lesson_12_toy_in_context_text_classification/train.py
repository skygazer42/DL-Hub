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
from .model import InContextTextClassifier, ModelConfig, classification_accuracy


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 12 (NLP): toy in-context text classification without gradient updates."
    )

    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--support-per-class", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        support_per_class=args.support_per_class,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def _run_epoch(*, model: InContextTextClassifier, loader, max_batches: int | None) -> float:
    total_examples = 0
    total_correct = 0.0

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= int(max_batches):
                break
            outputs = model(batch)
            labels = outputs["labels"]
            preds = outputs["predictions"]
            batch_acc = classification_accuracy(preds, labels)
            batch_examples = int(labels.numel())
            total_examples += batch_examples
            total_correct += batch_acc * batch_examples

    return total_correct / max(1, total_examples)


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="nlp",
        lesson="lesson_12_toy_in_context_text_classification",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("nlp.toy_in_context_text", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = InContextTextClassifier(ModelConfig(vocab_size=vocab.size, pad_id=vocab.pad_id))

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
            "notes": {"gradient_updates": False, "method": "support-query token overlap"},
        },
    )
    write_json(paths.run_dir / "vocab.json", vocab.to_dict())

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_acc = _run_epoch(model=model, loader=train_loader, max_batches=train_cfg.max_train_batches)
        eval_acc = _run_epoch(model=model, loader=val_loader, max_batches=train_cfg.max_eval_batches)
        logger.info(
            "Epoch %d/%d | train acc %.3f | eval acc %.3f",
            epoch,
            train_cfg.epochs,
            train_acc,
            eval_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_accuracy": train_acc,
                "eval_accuracy": eval_acc,
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        epoch=int(train_cfg.epochs),
        extra={
            "track": "nlp",
            "lesson": "lesson_12_toy_in_context_text_classification",
            "gradient_updates": False,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.nlp.lesson_12_toy_in_context_text_classification.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
