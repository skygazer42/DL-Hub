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
from .model import ModelConfig, ToyMaskedLanguageModel, masked_token_accuracy


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class EpochStats:
    loss: float
    masked_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 13 (NLP): toy masked language modeling with a tiny transformer encoder."
    )

    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mask_prob=args.mask_prob,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def _move_dict_to_device(d: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in d.items()}


def _run_epoch(
    *,
    model: ToyMaskedLanguageModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: torch.nn.Module,
    max_batches: int | None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_tokens = 0
    total_loss = 0.0
    total_correct = 0.0

    for step, (inputs, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        inputs = _move_dict_to_device(inputs, device)
        targets = _move_dict_to_device(targets, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        logits = outputs["logits"]
        labels = targets["labels"]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        if is_train:
            loss.backward()
            optimizer.step()

        masked_count = int(labels.ne(-100).sum().item())
        total_tokens += masked_count
        total_loss += float(loss.item()) * masked_count
        total_correct += masked_token_accuracy(logits.detach(), labels.detach()) * masked_count

    return EpochStats(
        loss=total_loss / max(1, total_tokens),
        masked_acc=total_correct / max(1, total_tokens),
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="nlp",
        lesson="lesson_13_toy_masked_language_modeling",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("nlp.toy_masked_lm", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = ToyMaskedLanguageModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=data_cfg.max_length,
            embed_dim=train_cfg.embed_dim,
            num_heads=train_cfg.num_heads,
            num_layers=train_cfg.num_layers,
            ff_dim=train_cfg.ff_dim,
            dropout=train_cfg.dropout,
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

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            criterion=criterion,
            max_batches=train_cfg.max_train_batches,
        )
        eval_stats = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            criterion=criterion,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info(
            "Epoch %d/%d | train loss %.4f masked_acc %.3f | eval loss %.4f masked_acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.masked_acc,
            eval_stats.loss,
            eval_stats.masked_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_masked_acc": train_stats.masked_acc,
                "eval_loss": eval_stats.loss,
                "eval_masked_acc": eval_stats.masked_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={
            "track": "nlp",
            "lesson": "lesson_13_toy_masked_language_modeling",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.nlp.lesson_13_toy_masked_language_modeling.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
