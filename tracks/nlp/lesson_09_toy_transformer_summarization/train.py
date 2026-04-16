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
from .model import ModelConfig, ToyTransformerSummarizer


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    embed_dim: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class Stats:
    loss: float
    token_acc: float
    exact_match: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 09 (NLP): toy encoder-decoder transformer summarization."
    )

    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-len", type=int, default=6)
    parser.add_argument("--max-len", type=int, default=18)
    parser.add_argument("--base-vocab-size", type=int, default=32)
    parser.add_argument("--summary-tokens", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
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
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        min_len=args.min_len,
        max_len=args.max_len,
        base_vocab_size=args.base_vocab_size,
        summary_tokens=args.summary_tokens,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
    )
    return train_cfg, data_cfg


def _run_epoch(
    *,
    model: ToyTransformerSummarizer,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pad_id: int,
    max_batches: int | None,
) -> Stats:
    is_train = optimizer is not None
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(pad_id))

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    total_seqs = 0
    exact = 0

    for step, (inputs, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        src_ids = inputs["src_ids"].to(device)
        src_mask = inputs["src_mask"].to(device)
        tgt_in_ids = inputs["tgt_in_ids"].to(device)
        tgt_mask = inputs["tgt_mask"].to(device)
        tgt_out_ids = targets["tgt_out_ids"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        out = model(src_ids=src_ids, src_mask=src_mask, tgt_in_ids=tgt_in_ids)
        logits = out["logits"]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out_ids.reshape(-1))

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = tgt_mask.to(torch.bool)
            correct_tokens += int(((pred == tgt_out_ids) & mask).sum().item())
            total_tokens += int(mask.sum().item())
            exact += int(((pred == tgt_out_ids) | (~mask)).all(dim=1).sum().item())
            batch_size = int(src_ids.shape[0])
            total_seqs += batch_size
            total_loss += float(loss.item()) * batch_size

    seq_denom = max(1, total_seqs)
    tok_denom = max(1, total_tokens)
    return Stats(
        loss=total_loss / seq_denom,
        token_acc=float(correct_tokens) / tok_denom,
        exact_match=float(exact) / seq_denom,
    )


def _write_samples(
    *,
    model: ToyTransformerSummarizer,
    vocab,
    device: torch.device,
    loader,
    out_path,
    epoch: int,
) -> None:
    model.eval()
    inputs, targets = next(iter(loader))
    src_ids = inputs["src_ids"][:8].to(device)
    src_mask = inputs["src_mask"][:8].to(device)
    tgt_out = targets["tgt_out_ids"][:8].tolist()

    pred = model.greedy_decode(
        src_ids=src_ids, src_mask=src_mask, max_len=int(inputs["tgt_in_ids"].shape[1])
    )
    rows = []
    for idx, pred_ids in enumerate(pred.cpu().tolist()):
        rows.append(
            {
                "epoch": int(epoch),
                "src": [
                    vocab.id_to_token(x) for x in src_ids[idx].cpu().tolist() if x != vocab.pad_id
                ],
                "tgt": [vocab.id_to_token(x) for x in tgt_out[idx] if x != vocab.pad_id],
                "pred": [vocab.id_to_token(x) for x in pred_ids if x != vocab.pad_id],
            }
        )

    with open(out_path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="nlp",
        lesson="lesson_09_toy_transformer_summarization",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("nlp.toy_transformer_summarization", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = ToyTransformerSummarizer(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_src_len=data_cfg.max_len,
            max_tgt_len=data_cfg.summary_tokens + 1,
            embed_dim=train_cfg.embed_dim,
            num_heads=train_cfg.num_heads,
            num_encoder_layers=train_cfg.num_encoder_layers,
            num_decoder_layers=train_cfg.num_decoder_layers,
            ff_dim=train_cfg.ff_dim,
            dropout=train_cfg.dropout,
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
        _write_samples(
            model=model,
            vocab=vocab,
            device=device_info.torch_device,
            loader=val_loader,
            out_path=samples_path,
            epoch=epoch,
        )
        logger.info(
            "Epoch %d/%d | train loss %.4f tok_acc %.3f EM %.3f | eval loss %.4f tok_acc %.3f EM %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.token_acc,
            train_stats.exact_match,
            eval_stats.loss,
            eval_stats.token_acc,
            eval_stats.exact_match,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_token_acc": train_stats.token_acc,
                "train_exact_match": train_stats.exact_match,
                "eval_loss": eval_stats.loss,
                "eval_token_acc": eval_stats.token_acc,
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
            "lesson": "lesson_09_toy_transformer_summarization",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.nlp.lesson_09_toy_transformer_summarization.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
