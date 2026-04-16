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

from .data import DataConfig, Vocab, get_dataloaders
from .model import ModelConfig, PreferenceTransformerLM


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    beta: float = 0.5
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
        description="Lesson 06 (LLM): toy pairwise preference optimization."
    )

    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=32)
    parser.add_argument("--base-vocab-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

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
        beta=args.beta,
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
        seq_length=args.seq_length,
        base_vocab_size=args.base_vocab_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=0,
    )
    return train_cfg, data_cfg


def _sequence_logprob(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    valid = labels.ne(int(ignore_index))
    safe_labels = labels.masked_fill(~valid, 0)
    token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * valid.to(torch.float32)
    lengths = valid.sum(dim=-1).clamp_min(1).to(torch.float32)
    return token_log_probs.sum(dim=-1) / lengths


def preference_dpo_loss(
    *,
    chosen_policy_logits: torch.Tensor,
    rejected_policy_logits: torch.Tensor,
    chosen_ref_logits: torch.Tensor,
    rejected_ref_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float,
    ignore_index: int,
) -> torch.Tensor:
    chosen_policy_lp = _sequence_logprob(chosen_policy_logits, chosen_labels, ignore_index)
    rejected_policy_lp = _sequence_logprob(rejected_policy_logits, rejected_labels, ignore_index)
    chosen_ref_lp = _sequence_logprob(chosen_ref_logits, chosen_labels, ignore_index)
    rejected_ref_lp = _sequence_logprob(rejected_ref_logits, rejected_labels, ignore_index)

    margin = (chosen_policy_lp - rejected_policy_lp) - (chosen_ref_lp - rejected_ref_lp)
    return -F.logsigmoid(float(beta) * margin).mean()


@torch.no_grad()
def _preference_margin(
    *,
    policy_model: PreferenceTransformerLM,
    reference_model: PreferenceTransformerLM,
    batch: dict[str, torch.Tensor],
    ignore_index: int,
    device: torch.device,
) -> torch.Tensor:
    chosen_inputs = {
        "input_ids": batch["chosen_input_ids"].to(device),
        "attention_mask": batch["chosen_attention_mask"].to(device),
    }
    rejected_inputs = {
        "input_ids": batch["rejected_input_ids"].to(device),
        "attention_mask": batch["rejected_attention_mask"].to(device),
    }
    chosen_labels = batch["chosen_labels"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)

    chosen_policy_lp = _sequence_logprob(policy_model(chosen_inputs), chosen_labels, ignore_index)
    rejected_policy_lp = _sequence_logprob(policy_model(rejected_inputs), rejected_labels, ignore_index)
    chosen_ref_lp = _sequence_logprob(reference_model(chosen_inputs), chosen_labels, ignore_index)
    rejected_ref_lp = _sequence_logprob(reference_model(rejected_inputs), rejected_labels, ignore_index)
    return (chosen_policy_lp - rejected_policy_lp) - (chosen_ref_lp - rejected_ref_lp)


def _step_loader(
    loader,
    *,
    policy_model: PreferenceTransformerLM,
    reference_model: PreferenceTransformerLM,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    beta: float,
    ignore_index: int,
    max_batches: int | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    policy_model.train(mode=is_train)
    reference_model.eval()

    total_loss = 0.0
    total_pref_acc = 0.0
    steps = 0
    for batch in loader:
        chosen_inputs = {
            "input_ids": batch["chosen_input_ids"].to(device),
            "attention_mask": batch["chosen_attention_mask"].to(device),
        }
        rejected_inputs = {
            "input_ids": batch["rejected_input_ids"].to(device),
            "attention_mask": batch["rejected_attention_mask"].to(device),
        }
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        chosen_policy_logits = policy_model(chosen_inputs)
        rejected_policy_logits = policy_model(rejected_inputs)
        with torch.no_grad():
            chosen_ref_logits = reference_model(chosen_inputs)
            rejected_ref_logits = reference_model(rejected_inputs)

        loss = preference_dpo_loss(
            chosen_policy_logits=chosen_policy_logits,
            rejected_policy_logits=rejected_policy_logits,
            chosen_ref_logits=chosen_ref_logits,
            rejected_ref_logits=rejected_ref_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            beta=float(beta),
            ignore_index=int(ignore_index),
        )
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            chosen_policy_lp = _sequence_logprob(chosen_policy_logits, chosen_labels, ignore_index)
            rejected_policy_lp = _sequence_logprob(rejected_policy_logits, rejected_labels, ignore_index)
            chosen_ref_lp = _sequence_logprob(chosen_ref_logits, chosen_labels, ignore_index)
            rejected_ref_lp = _sequence_logprob(rejected_ref_logits, rejected_labels, ignore_index)
            margin = (chosen_policy_lp - rejected_policy_lp) - (chosen_ref_lp - rejected_ref_lp)
            pref_acc = margin.gt(0).to(torch.float32).mean().item()

        total_loss += float(loss.item())
        total_pref_acc += float(pref_acc)
        steps += 1
        if max_batches is not None and steps >= int(max_batches):
            break

    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_pref_acc / steps


def _sample_prompt(vocab: Vocab) -> list[int]:
    topic_id = int(vocab.content_start_id)
    return [
        int(vocab.prompt_token_id),
        int(vocab.task_token_id),
        topic_id,
        int(vocab.separator_token_id),
    ]


def _write_samples(
    *,
    policy_model: PreferenceTransformerLM,
    reference_model: PreferenceTransformerLM,
    vocab: Vocab,
    out_path,
    epoch: int,
    device: torch.device,
) -> None:
    topic_id = int(vocab.content_start_id)
    c1 = int(vocab.content_start_id + ((topic_id - int(vocab.content_start_id) + 1) % int(vocab.base_vocab_size)))
    c2 = int(vocab.content_start_id + ((c1 - int(vocab.content_start_id) + 1) % int(vocab.base_vocab_size)))
    prompt = _sample_prompt(vocab)
    chosen = prompt + [int(vocab.chosen_token_id), topic_id, c1, c2, int(vocab.eos_id)]
    rejected = prompt + [int(vocab.rejected_token_id), c1, topic_id, c2, int(vocab.eos_id)]

    seq_len = int(policy_model.cfg.max_length)
    chosen_ids = chosen + [int(vocab.pad_id)] * max(0, seq_len - len(chosen))
    rejected_ids = rejected + [int(vocab.pad_id)] * max(0, seq_len - len(rejected))
    chosen_mask = [1.0] * min(len(chosen), seq_len) + [0.0] * max(0, seq_len - len(chosen))
    rejected_mask = [1.0] * min(len(rejected), seq_len) + [0.0] * max(0, seq_len - len(rejected))
    chosen_labels = [int(vocab.ignore_index)] * seq_len
    rejected_labels = [int(vocab.ignore_index)] * seq_len
    chosen_targets = [topic_id, c1, c2, int(vocab.eos_id)]
    rejected_targets = [c1, topic_id, c2, int(vocab.eos_id)]
    for i, target in enumerate(chosen_targets):
        pos = len(prompt) + i
        if pos < seq_len:
            chosen_labels[pos] = target
    for i, target in enumerate(rejected_targets):
        pos = len(prompt) + i
        if pos < seq_len:
            rejected_labels[pos] = target

    batch = {
        "chosen_input_ids": torch.tensor([chosen_ids], dtype=torch.long),
        "chosen_attention_mask": torch.tensor([chosen_mask], dtype=torch.float32),
        "chosen_labels": torch.tensor([chosen_labels], dtype=torch.long),
        "rejected_input_ids": torch.tensor([rejected_ids], dtype=torch.long),
        "rejected_attention_mask": torch.tensor([rejected_mask], dtype=torch.float32),
        "rejected_labels": torch.tensor([rejected_labels], dtype=torch.long),
    }
    margin = _preference_margin(
        policy_model=policy_model,
        reference_model=reference_model,
        batch=batch,
        ignore_index=int(vocab.ignore_index),
        device=device,
    )
    row = {
        "epoch": int(epoch),
        "prompt_ids": prompt,
        "chosen_ids": chosen,
        "rejected_ids": rejected,
        "margin": float(margin.item()),
    }
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="llm",
        lesson="lesson_06_toy_preference_optimization",
        run_name=train_cfg.run_name,
    )
    logger = get_logger(
        "llm.toy_preference_optimization",
        log_file=paths.logs_dir / "train.log",
    )
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
    policy_model = PreferenceTransformerLM(model_cfg).to(device_info.torch_device)
    reference_model = PreferenceTransformerLM(model_cfg).to(device_info.torch_device)
    reference_model.load_state_dict(policy_model.state_dict())
    for param in reference_model.parameters():
        param.requires_grad_(False)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )
    write_json(paths.run_dir / "vocab.json", vocab.to_dict())

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=float(train_cfg.learning_rate))
    metrics_path = paths.run_dir / "metrics.jsonl"
    samples_path = paths.run_dir / "samples.jsonl"

    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_loss, train_pref_acc = _step_loader(
            train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            optimizer=optimizer,
            device=device_info.torch_device,
            beta=float(train_cfg.beta),
            ignore_index=int(vocab.ignore_index),
            max_batches=train_cfg.max_train_batches,
        )
        eval_loss, eval_pref_acc = _step_loader(
            val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            optimizer=None,
            device=device_info.torch_device,
            beta=float(train_cfg.beta),
            ignore_index=int(vocab.ignore_index),
            max_batches=train_cfg.max_eval_batches,
        )
        _write_samples(
            policy_model=policy_model,
            reference_model=reference_model,
            vocab=vocab,
            out_path=samples_path,
            epoch=epoch,
            device=device_info.torch_device,
        )

        logger.info(
            "Epoch %d/%d | train loss %.4f pref_acc %.3f | eval loss %.4f pref_acc %.3f",
            epoch,
            train_cfg.epochs,
            train_loss,
            train_pref_acc,
            eval_loss,
            eval_pref_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pref_acc": train_pref_acc,
                "eval_loss": eval_loss,
                "eval_pref_acc": eval_pref_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=policy_model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={
            "track": "llm",
            "lesson": "lesson_06_toy_preference_optimization",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.llm.lesson_06_toy_preference_optimization.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
