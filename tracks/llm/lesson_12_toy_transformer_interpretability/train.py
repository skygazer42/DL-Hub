import argparse
import json
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
from .model import ModelConfig, ToyInterpretabilityTransformerLM


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
    ff_dim: int = 256
    dropout: float = 0.1


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 12 (LLM): toy transformer interpretability with attention and saliency."
    )
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=16)
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
        ff_dim=args.ff_dim,
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


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, vocab_size = logits.shape
    loss = F.cross_entropy(
        logits.reshape(bsz * seq_len, vocab_size),
        labels.reshape(bsz * seq_len),
        ignore_index=int(ignore_index),
    )
    pred = logits.argmax(dim=-1)
    valid = labels.ne(int(ignore_index))
    correct = ((pred == labels) & valid).to(torch.float32).sum()
    total = valid.to(torch.float32).sum().clamp_min(1.0)
    acc = correct / total
    return loss, acc


@torch.no_grad()
def compute_attention_map(
    model: ToyInterpretabilityTransformerLM, inputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    _, attn = model(inputs, return_attention=True)
    return attn


def compute_token_saliency(
    *,
    model: ToyInterpretabilityTransformerLM,
    inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    input_ids = inputs["input_ids"].to(torch.long)
    attention_mask = inputs["attention_mask"].to(torch.float32)
    embeddings = model.embed_inputs(input_ids).detach().requires_grad_(True)
    logits = model.forward_from_embeddings(embeddings, attention_mask)

    log_probs = torch.log_softmax(logits, dim=-1)
    valid = labels.ne(int(ignore_index))
    safe_labels = labels.masked_fill(~valid, 0)
    token_lp = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    loss = -((token_lp * valid.to(torch.float32)).sum() / valid.to(torch.float32).sum().clamp_min(1.0))
    loss.backward()

    saliency = embeddings.grad.norm(dim=-1)
    saliency = saliency / saliency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return saliency.detach()


def _run_epoch(
    *,
    model: ToyInterpretabilityTransformerLM,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    ignore_index: int,
    max_batches: int | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = _move_batch(batch, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(
            {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        )
        loss, acc = _token_cross_entropy(logits, batch["labels"], ignore_index=ignore_index)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += float(acc.item())
        steps += 1

    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_acc / steps


def _write_samples(
    *,
    model: ToyInterpretabilityTransformerLM,
    loader,
    out_path,
    epoch: int,
    ignore_index: int,
    device: torch.device,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    batch = _move_batch(batch, device)
    inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    with torch.no_grad():
        attention = compute_attention_map(model, inputs)
    saliency = compute_token_saliency(
        model=model,
        inputs=inputs,
        labels=batch["labels"],
        ignore_index=int(ignore_index),
    )

    row = {
        "epoch": int(epoch),
        "input_ids": batch["input_ids"][0].detach().cpu().tolist(),
        "attention_head0_last_query": attention[0, 0, -1].detach().cpu().tolist(),
        "saliency": saliency[0].detach().cpu().tolist(),
    }
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="llm",
        lesson="lesson_12_toy_transformer_interpretability",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("llm.toy_transformer_interpretability", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = ToyInterpretabilityTransformerLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=int(data_cfg.seq_length),
            embed_dim=int(train_cfg.embed_dim),
            num_heads=int(train_cfg.num_heads),
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
    samples_path = paths.run_dir / "samples.jsonl"

    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_loss, train_acc = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device_info.torch_device,
            ignore_index=int(vocab.ignore_index),
            max_batches=train_cfg.max_train_batches,
        )
        with torch.no_grad():
            eval_loss, eval_acc = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device_info.torch_device,
                ignore_index=int(vocab.ignore_index),
                max_batches=train_cfg.max_eval_batches,
            )
        _write_samples(
            model=model,
            loader=val_loader,
            out_path=samples_path,
            epoch=epoch,
            ignore_index=int(vocab.ignore_index),
            device=device_info.torch_device,
        )
        logger.info(
            "Epoch %d/%d | train loss %.4f acc %.3f | eval loss %.4f acc %.3f",
            epoch,
            train_cfg.epochs,
            train_loss,
            train_acc,
            eval_loss,
            eval_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
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
            "lesson": "lesson_12_toy_transformer_interpretability",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.llm.lesson_12_toy_transformer_interpretability.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
