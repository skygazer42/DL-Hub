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
from .model import ModelConfig, CompactReplacedTokenDetectionTransformer


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    rtd_loss_weight: float = 0.5
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
        description="Lesson 14 (LLM): compact replaced-token detection transformer pretraining."
    )
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=18)
    parser.add_argument("--base-vocab-size", type=int, default=64)
    parser.add_argument("--replace-probability", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--rtd-loss-weight", type=float, default=0.5)
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
        rtd_loss_weight=args.rtd_loss_weight,
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
        replace_probability=args.replace_probability,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def replaced_token_detection_loss(
    *,
    token_logits: torch.Tensor,
    rtd_logits: torch.Tensor,
    labels: torch.Tensor,
    replaced_labels: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_index: int,
    rtd_loss_weight: float,
) -> torch.Tensor:
    bsz, seq_len, vocab_size = token_logits.shape
    token_loss = F.cross_entropy(
        token_logits.reshape(bsz * seq_len, vocab_size),
        labels.reshape(bsz * seq_len),
        ignore_index=int(ignore_index),
    )

    token_mask = attention_mask.to(torch.float32)
    rtd_per_token = F.binary_cross_entropy_with_logits(
        rtd_logits,
        replaced_labels.to(torch.float32),
        reduction="none",
    )
    rtd_loss = (rtd_per_token * token_mask).sum() / token_mask.sum().clamp_min(1.0)
    return token_loss + float(rtd_loss_weight) * rtd_loss


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _run_epoch(
    *,
    model: CompactReplacedTokenDetectionTransformer,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ignore_index: int,
    rtd_loss_weight: float,
    max_batches: int | None,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_token_acc = 0.0
    total_rtd_acc = 0.0
    steps = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
        loss = replaced_token_detection_loss(
            token_logits=outputs["token_logits"],
            rtd_logits=outputs["rtd_logits"],
            labels=batch["labels"],
            replaced_labels=batch["replaced_labels"],
            attention_mask=batch["attention_mask"],
            ignore_index=int(ignore_index),
            rtd_loss_weight=float(rtd_loss_weight),
        )
        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            token_pred = outputs["token_logits"].argmax(dim=-1)
            valid = batch["labels"].ne(int(ignore_index))
            token_correct = ((token_pred == batch["labels"]) & valid).to(torch.float32).sum()
            token_count = valid.to(torch.float32).sum().clamp_min(1.0)
            token_acc = token_correct / token_count

            rtd_pred = (torch.sigmoid(outputs["rtd_logits"]) >= 0.5).to(torch.float32)
            mask = batch["attention_mask"].to(torch.float32)
            rtd_correct = ((rtd_pred == batch["replaced_labels"]).to(torch.float32) * mask).sum()
            rtd_count = mask.sum().clamp_min(1.0)
            rtd_acc = rtd_correct / rtd_count

        total_loss += float(loss.item())
        total_token_acc += float(token_acc.item())
        total_rtd_acc += float(rtd_acc.item())
        steps += 1

    if steps == 0:
        return 0.0, 0.0, 0.0
    return total_loss / steps, total_token_acc / steps, total_rtd_acc / steps


def _write_samples(
    *,
    model: CompactReplacedTokenDetectionTransformer,
    loader,
    out_path,
    epoch: int,
    device: torch.device,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    batch = _move_batch(batch, device)
    with torch.no_grad():
        outputs = model({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
        rtd_prob = torch.sigmoid(outputs["rtd_logits"])

    row = {
        "epoch": int(epoch),
        "input_ids": batch["input_ids"][0].detach().cpu().tolist(),
        "replaced_labels": batch["replaced_labels"][0].detach().cpu().tolist(),
        "rtd_probability": rtd_prob[0].detach().cpu().tolist(),
    }
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="llm",
        lesson="lesson_14_compact_replaced_token_detection_transformer",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("llm.compact_replaced_token_detection", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = CompactReplacedTokenDetectionTransformer(
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
        train_loss, train_token_acc, train_rtd_acc = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            ignore_index=int(vocab.ignore_index),
            rtd_loss_weight=float(train_cfg.rtd_loss_weight),
            max_batches=train_cfg.max_train_batches,
        )
        with torch.no_grad():
            eval_loss, eval_token_acc, eval_rtd_acc = _run_epoch(
                model=model,
                loader=val_loader,
                device=device_info.torch_device,
                optimizer=None,
                ignore_index=int(vocab.ignore_index),
                rtd_loss_weight=float(train_cfg.rtd_loss_weight),
                max_batches=train_cfg.max_eval_batches,
            )
        _write_samples(
            model=model,
            loader=val_loader,
            out_path=samples_path,
            epoch=epoch,
            device=device_info.torch_device,
        )
        logger.info(
            (
                "Epoch %d/%d | train loss %.4f tok_acc %.3f rtd_acc %.3f | "
                "eval loss %.4f tok_acc %.3f rtd_acc %.3f"
            ),
            epoch,
            train_cfg.epochs,
            train_loss,
            train_token_acc,
            train_rtd_acc,
            eval_loss,
            eval_token_acc,
            eval_rtd_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "train_token_acc": train_token_acc,
                "train_rtd_acc": train_rtd_acc,
                "eval_loss": eval_loss,
                "eval_token_acc": eval_token_acc,
                "eval_rtd_acc": eval_rtd_acc,
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
            "lesson": "lesson_14_compact_replaced_token_detection_transformer",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.llm.lesson_14_compact_replaced_token_detection_transformer.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
