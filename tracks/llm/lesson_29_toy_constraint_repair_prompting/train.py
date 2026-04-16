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
from dlhub.training.loop import evaluate_token_classifier, fit_token_classifier

from .data import DataConfig, Vocab, get_dataloaders
from .model import ConstraintRepairPromptingTransformerLM, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    generation_tokens: int = 8
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 29 (LLM): toy constraint-repair prompting with candidate repair spans."
    )
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=32)
    parser.add_argument("--base-vocab-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--generation-tokens", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
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
        generation_tokens=args.generation_tokens,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
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


@torch.no_grad()
def generate_response(
    *,
    model: ConstraintRepairPromptingTransformerLM,
    device: torch.device,
    prompt_ids: list[int],
    max_new_tokens: int,
    pad_id: int,
    stop_id: int,
) -> list[int]:
    model.eval()
    max_length = int(model.cfg.max_length)
    generated = prompt_ids[:]
    for _ in range(int(max_new_tokens)):
        cur = generated[-max_length:]
        padded = cur + [int(pad_id)] * max(0, max_length - len(cur))
        mask = [1.0] * min(len(cur), max_length) + [0.0] * max(0, max_length - len(cur))
        inputs = {
            "input_ids": torch.tensor([padded], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([mask], dtype=torch.float32, device=device),
        }
        logits = model(inputs)
        next_pos = min(len(cur), max_length) - 1
        next_id = int(logits[0, next_pos].argmax(dim=-1).item())
        generated.append(next_id)
        if next_id == int(stop_id):
            break
    return generated


def _sample_prompt(vocab: Vocab) -> list[int]:
    topic_id = int(vocab.content_start_id)
    return [
        int(vocab.prompt_token_id),
        topic_id,
        int(vocab.separator_token_id),
        int(vocab.constraint_token_id),
        int(topic_id + 1),
        int(topic_id + 3),
        int(vocab.separator_token_id),
        int(vocab.candidate_token_id),
        int(topic_id + 5),
        int(topic_id + 2),
        int(vocab.separator_token_id),
        int(vocab.violates_token_id),
        int(vocab.separator_token_id),
        int(vocab.repair_token_id),
    ]


def _write_samples(
    *,
    model: ConstraintRepairPromptingTransformerLM,
    device: torch.device,
    vocab: Vocab,
    out_path,
    epoch: int,
    generation_tokens: int,
) -> None:
    prompt_ids = _sample_prompt(vocab)
    generated = generate_response(
        model=model,
        device=device,
        prompt_ids=prompt_ids,
        max_new_tokens=int(generation_tokens),
        pad_id=vocab.pad_id,
        stop_id=vocab.eos_id,
    )
    row = {"epoch": int(epoch), "prompt_ids": prompt_ids, "gen_ids": generated}
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="llm",
        lesson="lesson_29_toy_constraint_repair_prompting",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("llm.toy_constraint_repair_prompting", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = ConstraintRepairPromptingTransformerLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=int(data_cfg.seq_length),
            embed_dim=int(train_cfg.embed_dim),
            num_heads=int(train_cfg.num_heads),
            num_layers=int(train_cfg.num_layers),
            ff_dim=int(train_cfg.ff_dim),
            dropout=float(train_cfg.dropout),
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

    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(vocab.ignore_index))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    metrics_path = paths.run_dir / "metrics.jsonl"
    samples_path = paths.run_dir / "samples.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = fit_token_classifier(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device_info.torch_device,
            max_batches=train_cfg.max_train_batches,
            ignore_index=int(vocab.ignore_index),
        )
        eval_stats = evaluate_token_classifier(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
            ignore_index=int(vocab.ignore_index),
        )
        _write_samples(
            model=model,
            device=device_info.torch_device,
            vocab=vocab,
            out_path=samples_path,
            epoch=epoch,
            generation_tokens=int(train_cfg.generation_tokens),
        )
        logger.info(
            "Epoch %d/%d | train loss %.4f acc %.3f | eval loss %.4f acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.accuracy,
            eval_stats.loss,
            eval_stats.accuracy,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_acc": train_stats.accuracy,
                "eval_loss": eval_stats.loss,
                "eval_acc": eval_stats.accuracy,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={
            "track": "llm",
            "lesson": "lesson_29_toy_constraint_repair_prompting",
            "vocab_size": vocab.size,
        },
    )
    return 0


def main() -> int:
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
