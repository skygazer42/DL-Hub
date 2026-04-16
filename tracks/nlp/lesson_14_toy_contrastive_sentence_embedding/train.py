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
from .model import ContrastiveSentenceEncoder, ModelConfig, contrastive_accuracy, nt_xent_loss


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
    proj_dim: int = 32
    dropout: float = 0.1
    temperature: float = 0.1


@dataclass(frozen=True)
class EpochStats:
    loss: float
    contrastive_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 14 (NLP): toy contrastive sentence embeddings with two augmented views."
    )

    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--dropout-prob", type=float, default=0.2)
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
    parser.add_argument("--proj-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
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
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        temperature=args.temperature,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dropout_prob=args.dropout_prob,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def _move_dict_to_device(d: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in d.items()}


def _run_epoch(
    *,
    model: ContrastiveSentenceEncoder,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    temperature: float,
    max_batches: int | None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_examples = 0
    total_loss = 0.0
    total_acc = 0.0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        batch = _move_dict_to_device(batch, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(batch)
        loss = nt_xent_loss(outputs["sim_matrix"], temperature=float(temperature))

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(outputs["sim_matrix"].shape[0])
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_acc += contrastive_accuracy(outputs["sim_matrix"].detach()) * batch_size

    return EpochStats(
        loss=total_loss / max(1, total_examples),
        contrastive_acc=total_acc / max(1, total_examples),
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="nlp",
        lesson="lesson_14_toy_contrastive_sentence_embedding",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("nlp.toy_contrastive_sentence_embedding", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = ContrastiveSentenceEncoder(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=train_cfg.embed_dim,
            proj_dim=train_cfg.proj_dim,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            temperature=train_cfg.temperature,
            max_batches=train_cfg.max_train_batches,
        )
        eval_stats = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            temperature=train_cfg.temperature,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info(
            "Epoch %d/%d | train loss %.4f contrastive_acc %.3f | eval loss %.4f contrastive_acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.contrastive_acc,
            eval_stats.loss,
            eval_stats.contrastive_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_contrastive_acc": train_stats.contrastive_acc,
                "eval_loss": eval_stats.loss,
                "eval_contrastive_acc": eval_stats.contrastive_acc,
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
            "lesson": "lesson_14_toy_contrastive_sentence_embedding",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.nlp.lesson_14_toy_contrastive_sentence_embedding.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
