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
from .model import ModelConfig, SimpleSpanQA


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 3e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"

    embed_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.1


@dataclass(frozen=True)
class Stats:
    loss: float
    start_acc: float
    end_acc: float
    exact_match: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 07 (NLP): compact reading comprehension (span prediction)."
    )

    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--question-length", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
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
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        context_length=args.context_length,
        question_length=args.question_length,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def _run_epoch(
    *,
    model: SimpleSpanQA,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> Stats:
    is_train = optimizer is not None
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    start_correct = 0
    end_correct = 0
    em_correct = 0

    if is_train:
        model.train()
    else:
        model.eval()

    for step, (inputs, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        inputs = {k: v.to(device) for k, v in inputs.items()}
        start = targets["start"].to(device)
        end = targets["end"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        out = model(**inputs)
        loss_start = criterion(out["start_logits"], start)
        loss_end = criterion(out["end_logits"], end)
        loss = 0.5 * (loss_start + loss_end)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_s = out["start_logits"].argmax(dim=1)
            pred_e = out["end_logits"].argmax(dim=1)
            bsz = int(start.shape[0])
            total += bsz
            start_correct += int((pred_s == start).sum().item())
            end_correct += int((pred_e == end).sum().item())
            em_correct += int(((pred_s == start) & (pred_e == end)).sum().item())
            total_loss += float(loss.item()) * bsz

    return Stats(
        loss=total_loss / max(1, total),
        start_acc=float(start_correct) / max(1, total),
        end_acc=float(end_correct) / max(1, total),
        exact_match=float(em_correct) / max(1, total),
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="nlp", lesson="lesson_07_reading_comprehension", run_name=train_cfg.run_name
    )
    logger = get_logger("nlp.synthetic_rc", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = SimpleSpanQA(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=train_cfg.embed_dim,
            hidden_dim=train_cfg.hidden_dim,
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
            "Epoch %d/%d | train loss %.4f EM %.3f | eval loss %.4f EM %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.exact_match,
            eval_stats.loss,
            eval_stats.exact_match,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_start_acc": train_stats.start_acc,
                "train_end_acc": train_stats.end_acc,
                "train_exact_match": train_stats.exact_match,
                "eval_loss": eval_stats.loss,
                "eval_start_acc": eval_stats.start_acc,
                "eval_end_acc": eval_stats.end_acc,
                "eval_exact_match": eval_stats.exact_match,
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
            "lesson": "lesson_07_reading_comprehension",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.nlp.lesson_07_reading_comprehension.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
