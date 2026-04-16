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
from .model import ModelConfig, ToyLlmJudge


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    judge_loss_weight: float = 0.5
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    embed_dim: int = 128
    hidden_dim: int = 192
    dropout: float = 0.1


@dataclass(frozen=True)
class EpochStats:
    loss: float
    token_accuracy: float
    judge_accuracy: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 15 (LLM): toy LLM judge that scores candidate answers."
    )
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--base-vocab-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--judge-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        judge_loss_weight=args.judge_loss_weight,
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
        seq_length=args.seq_length,
        base_vocab_size=args.base_vocab_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def llm_judge_loss(
    *,
    token_logits: torch.Tensor,
    judge_logits: torch.Tensor,
    labels: torch.Tensor,
    judge_targets: torch.Tensor,
    ignore_index: int,
    judge_loss_weight: float,
) -> torch.Tensor:
    b, t, v = token_logits.shape
    lm_loss = torch.nn.CrossEntropyLoss(ignore_index=int(ignore_index))(
        token_logits.reshape(b * t, v),
        labels.reshape(b * t),
    )
    judge_loss = torch.nn.BCEWithLogitsLoss()(
        judge_logits.to(torch.float32),
        judge_targets.to(torch.float32),
    )
    return lm_loss + float(judge_loss_weight) * judge_loss


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _run_epoch(
    *,
    model: ToyLlmJudge,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ignore_index: int,
    judge_loss_weight: float,
    max_batches: int | None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_token_count = 0
    correct_token_count = 0
    total_examples = 0
    correct_judges = 0
    steps = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        batch = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = llm_judge_loss(
            token_logits=outputs["token_logits"],
            judge_logits=outputs["judge_logits"],
            labels=batch["labels"],
            judge_targets=batch["judge_targets"],
            ignore_index=int(ignore_index),
            judge_loss_weight=float(judge_loss_weight),
        )
        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            token_mask = batch["labels"] != int(ignore_index)
            n_tokens = int(token_mask.sum().item())
            if n_tokens > 0:
                token_pred = outputs["token_logits"].argmax(dim=-1)
                token_correct = int(((token_pred == batch["labels"]) & token_mask).sum().item())
                correct_token_count += token_correct
                total_token_count += n_tokens

            judge_pred = (torch.sigmoid(outputs["judge_logits"]) >= 0.5).to(torch.float32)
            judge_correct = (judge_pred == batch["judge_targets"].to(torch.float32)).sum().item()
            correct_judges += int(judge_correct)
            total_examples += int(batch["judge_targets"].shape[0])
            total_loss += float(loss.detach().item())
            steps += 1

    token_acc = 0.0 if total_token_count == 0 else (correct_token_count / total_token_count)
    judge_acc = 0.0 if total_examples == 0 else (correct_judges / total_examples)
    avg_loss = 0.0 if steps == 0 else (total_loss / steps)
    return EpochStats(loss=avg_loss, token_accuracy=token_acc, judge_accuracy=judge_acc)


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="llm",
        lesson="lesson_15_toy_llm_judge",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("llm.toy_llm_judge", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = ToyLlmJudge(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=int(data_cfg.seq_length),
            embed_dim=int(train_cfg.embed_dim),
            hidden_dim=int(train_cfg.hidden_dim),
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
            ignore_index=int(vocab.ignore_index),
            judge_loss_weight=float(train_cfg.judge_loss_weight),
            max_batches=train_cfg.max_train_batches,
        )
        with torch.no_grad():
            eval_stats = _run_epoch(
                model=model,
                loader=val_loader,
                device=device_info.torch_device,
                optimizer=None,
                ignore_index=int(vocab.ignore_index),
                judge_loss_weight=float(train_cfg.judge_loss_weight),
                max_batches=train_cfg.max_eval_batches,
            )
        logger.info(
            (
                "Epoch %d/%d | train loss %.4f tok_acc %.3f judge_acc %.3f | "
                "eval loss %.4f tok_acc %.3f judge_acc %.3f"
            ),
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.token_accuracy,
            train_stats.judge_accuracy,
            eval_stats.loss,
            eval_stats.token_accuracy,
            eval_stats.judge_accuracy,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": int(epoch),
                "train_loss": train_stats.loss,
                "train_token_acc": train_stats.token_accuracy,
                "train_judge_acc": train_stats.judge_accuracy,
                "eval_loss": eval_stats.loss,
                "eval_token_acc": eval_stats.token_accuracy,
                "eval_judge_acc": eval_stats.judge_accuracy,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={
            "track": "llm",
            "lesson": "lesson_15_toy_llm_judge",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.llm.lesson_15_toy_llm_judge.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
