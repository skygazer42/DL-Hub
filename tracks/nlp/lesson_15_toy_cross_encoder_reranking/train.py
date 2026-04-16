from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dlhub.checkpoint import save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed

from .data import DataConfig, get_dataloaders
from .model import CrossEncoderReranker, ModelConfig, reranking_accuracy


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class Stats:
    loss: float
    rerank_acc: float


def pairwise_ranking_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(positive_scores - negative_scores)).mean()


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 15 (NLP): toy cross-encoder reranking for query/document pairs."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-query-length", type=int, default=8)
    parser.add_argument("--max-doc-length", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
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
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_query_length=args.max_query_length,
        max_doc_length=args.max_doc_length,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _run_epoch(
    *,
    model: CrossEncoderReranker,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> Stats:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        batch = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(batch)
        loss = pairwise_ranking_loss(outputs["positive_scores"], outputs["negative_scores"])
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += reranking_accuracy(
            outputs["positive_scores"].detach(),
            outputs["negative_scores"].detach(),
        )
        total_steps += 1

    if total_steps == 0:
        return Stats(loss=0.0, rerank_acc=0.0)
    return Stats(loss=(total_loss / total_steps), rerank_acc=(total_acc / total_steps))


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="nlp",
        lesson="lesson_15_toy_cross_encoder_reranking",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("nlp.toy_cross_encoder_reranking", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = CrossEncoderReranker(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=int(data_cfg.max_query_length + data_cfg.max_doc_length + 1),
            embed_dim=int(train_cfg.embed_dim),
            num_heads=int(train_cfg.num_heads),
            num_layers=int(train_cfg.num_layers),
            ff_dim=int(train_cfg.ff_dim),
            dropout=float(train_cfg.dropout),
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
            "Epoch %d/%d | train loss %.4f rerank_acc %.3f | eval loss %.4f rerank_acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.rerank_acc,
            eval_stats.loss,
            eval_stats.rerank_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_rerank_acc": train_stats.rerank_acc,
                "eval_loss": eval_stats.loss,
                "eval_rerank_acc": eval_stats.rerank_acc,
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
            "lesson": "lesson_15_toy_cross_encoder_reranking",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.nlp.lesson_15_toy_cross_encoder_reranking.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
