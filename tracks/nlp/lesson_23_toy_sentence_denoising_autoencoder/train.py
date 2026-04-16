from __future__ import annotations

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

from .data import DataConfig, Vocab, get_dataloaders
from .model import DenoisingSeq2Seq, ModelConfig, reconstruction_token_accuracy


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    embed_dim: int = 32
    hidden_dim: int = 48
    dropout: float = 0.1


@dataclass(frozen=True)
class EpochStats:
    loss: float
    token_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 23 (NLP): toy sentence denoising autoencoder."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--corruption-prob", type=float, default=0.35)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=48)
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
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        corruption_prob=args.corruption_prob,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def _move_dict(d: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in d.items()}


def _run_epoch(
    *,
    model: DenoisingSeq2Seq,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pad_id: int,
    max_batches: int | None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(mode=is_train)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(pad_id))

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for step, (inputs, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break
        inputs = _move_dict(inputs, device)
        targets = _move_dict(targets, device)
        outputs = model(
            src_ids=inputs["src_ids"],
            src_mask=inputs["src_mask"],
            tgt_in_ids=inputs["tgt_in_ids"],
        )
        logits = outputs["logits"]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets["tgt_out_ids"].reshape(-1))
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += reconstruction_token_accuracy(logits.detach(), targets["tgt_out_ids"], int(pad_id))
        total_batches += 1

    return EpochStats(
        loss=total_loss / max(1, total_batches),
        token_acc=total_acc / max(1, total_batches),
    )


def _decode(ids: list[int], vocab: Vocab) -> list[str]:
    ignore = {vocab.pad_id, vocab.bos_id, vocab.eos_id}
    return [vocab.id_to_token[idx] for idx in ids if idx not in ignore]


def _write_samples(
    *,
    model: DenoisingSeq2Seq,
    vocab: Vocab,
    loader,
    device: torch.device,
    output_path,
) -> None:
    model.eval()
    inputs, targets = next(iter(loader))
    src_ids = inputs["src_ids"][:8].to(device)
    src_mask = inputs["src_mask"][:8].to(device)
    preds = model.greedy_decode(src_ids=src_ids, src_mask=src_mask, max_len=int(inputs["tgt_in_ids"].shape[1]))
    rows = []
    for idx in range(int(preds.shape[0])):
        rows.append(
            {
                "source": _decode(inputs["src_ids"][idx].tolist(), vocab),
                "target": _decode(targets["tgt_out_ids"][idx].tolist(), vocab),
                "prediction": _decode(preds[idx].cpu().tolist(), vocab),
            }
        )
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="nlp",
        lesson="lesson_23_toy_sentence_denoising_autoencoder",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("nlp.toy_sentence_denoising", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = DenoisingSeq2Seq(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
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
    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            pad_id=vocab.pad_id,
            max_batches=train_cfg.max_train_batches,
        )
        eval_stats = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            pad_id=vocab.pad_id,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info(
            "Epoch %d/%d | train loss %.4f token_acc %.3f | eval loss %.4f token_acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.token_acc,
            eval_stats.loss,
            eval_stats.token_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_token_acc": train_stats.token_acc,
                "eval_loss": eval_stats.loss,
                "eval_token_acc": eval_stats.token_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    _write_samples(
        model=model,
        vocab=vocab,
        loader=val_loader,
        device=device_info.torch_device,
        output_path=paths.run_dir / "samples.jsonl",
    )
    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "nlp", "lesson": "lesson_23_toy_sentence_denoising_autoencoder"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.nlp.lesson_23_toy_sentence_denoising_autoencoder.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
