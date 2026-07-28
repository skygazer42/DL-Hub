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

from .data import DataConfig, get_dataloaders
from .model import ModelConfig, CompactGrpoPolicyLM


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 10 (LLM): compact GRPO group-relative policy alignment."
    )

    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--base-vocab-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--embed-dim", type=int, default=128)
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
        num_prompts=args.num_prompts,
        group_size=args.group_size,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        base_vocab_size=args.base_vocab_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def _token_log_probs(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    valid = labels.ne(int(ignore_index))
    safe_labels = labels.masked_fill(~valid, 0)
    token_lp = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * valid.to(torch.float32)
    return token_lp, valid.to(torch.float32)


def grpo_group_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    group_rewards: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    token_lp, valid = _token_log_probs(logits, labels, ignore_index)
    token_mask = valid * response_mask.to(torch.float32)
    lengths = token_mask.sum(dim=-1).clamp_min(1.0)
    seq_lp = (token_lp * token_mask).sum(dim=-1) / lengths  # (B, G)

    rewards = group_rewards.to(torch.float32)
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    normalized = centered / (centered.std(dim=1, keepdim=True, unbiased=False) + 1e-6)

    # Maximize weighted log-prob for above-average candidates, minimize for below-average.
    return -(normalized * seq_lp).mean()


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _run_epoch(
    *,
    policy: CompactGrpoPolicyLM,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    ignore_index: int,
    max_batches: int | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    policy.train(mode=is_train)

    total_loss = 0.0
    total_adv_gap = 0.0
    steps = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = _move_batch(batch, device)
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = policy(model_inputs)
        loss = grpo_group_loss(
            logits=logits,
            labels=batch["labels"],
            response_mask=batch["response_mask"],
            group_rewards=batch["group_rewards"],
            ignore_index=int(ignore_index),
        )
        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            token_lp, valid = _token_log_probs(logits, batch["labels"], int(ignore_index))
            token_mask = valid * batch["response_mask"].to(torch.float32)
            lengths = token_mask.sum(dim=-1).clamp_min(1.0)
            seq_lp = (token_lp * token_mask).sum(dim=-1) / lengths
            rewards = batch["group_rewards"].to(torch.float32)
            best_idx = rewards.argmax(dim=1, keepdim=True)
            worst_idx = rewards.argmin(dim=1, keepdim=True)
            best_lp = seq_lp.gather(dim=1, index=best_idx)
            worst_lp = seq_lp.gather(dim=1, index=worst_idx)
            adv_gap = (best_lp - worst_lp).mean().item()

        total_loss += float(loss.item())
        total_adv_gap += float(adv_gap)
        steps += 1

    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_adv_gap / steps


def _write_samples(*, loader, out_path, epoch: int) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    row = {
        "epoch": int(epoch),
        "input_ids_group0": batch["input_ids"][0].detach().cpu().tolist(),
        "rewards_group0": [float(v) for v in batch["group_rewards"][0].detach().cpu().tolist()],
    }
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="llm",
        lesson="lesson_10_compact_grpo_alignment",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("llm.compact_grpo_alignment", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model_cfg = ModelConfig(
        vocab_size=vocab.size,
        pad_id=vocab.pad_id,
        max_length=int(data_cfg.seq_length),
        embed_dim=int(train_cfg.embed_dim),
        num_heads=int(train_cfg.num_heads),
        num_layers=int(train_cfg.num_layers),
        ff_dim=int(train_cfg.ff_dim),
        dropout=float(train_cfg.dropout),
    )
    policy = CompactGrpoPolicyLM(model_cfg).to(device_info.torch_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.learning_rate))

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
    samples_path = paths.run_dir / "samples.jsonl"

    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_loss, train_adv_gap = _run_epoch(
            policy=policy,
            loader=train_loader,
            optimizer=optimizer,
            device=device_info.torch_device,
            ignore_index=int(vocab.ignore_index),
            max_batches=train_cfg.max_train_batches,
        )
        with torch.no_grad():
            eval_loss, eval_adv_gap = _run_epoch(
                policy=policy,
                loader=val_loader,
                optimizer=None,
                device=device_info.torch_device,
                ignore_index=int(vocab.ignore_index),
                max_batches=train_cfg.max_eval_batches,
            )

        _write_samples(loader=val_loader, out_path=samples_path, epoch=epoch)
        logger.info(
            "Epoch %d/%d | train loss %.4f | eval loss %.4f | train gap %.4f | eval gap %.4f",
            epoch,
            train_cfg.epochs,
            train_loss,
            eval_loss,
            train_adv_gap,
            eval_adv_gap,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "train_adv_gap": train_adv_gap,
                "eval_adv_gap": eval_adv_gap,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=policy,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "llm", "lesson": "lesson_10_compact_grpo_alignment", "vocab_size": vocab.size},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.llm.lesson_10_compact_grpo_alignment.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
