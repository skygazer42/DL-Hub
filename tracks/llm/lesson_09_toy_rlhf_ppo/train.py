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
from .model import ModelConfig, ToyPolicyLM, ToyTokenRewardModel


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.05
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
    parser = argparse.ArgumentParser(description="Lesson 09 (LLM): toy RLHF PPO policy fine-tuning.")

    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--base-vocab-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-coefficient", type=float, default=0.05)
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
        clip_epsilon=args.clip_epsilon,
        kl_coefficient=args.kl_coefficient,
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


def _token_log_probs(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    valid = labels.ne(int(ignore_index))
    safe_labels = labels.masked_fill(~valid, 0)
    token_lp = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * valid.to(torch.float32)
    return token_lp, valid.to(torch.float32)


def ppo_policy_loss(
    *,
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    rewards: torch.Tensor,
    clip_epsilon: float,
    kl_coefficient: float,
    ignore_index: int,
) -> torch.Tensor:
    # Toy simplification: the "old policy" in the ratio is the frozen
    # initial reference model and the batch comes from a fixed dataset,
    # not on-policy rollouts. This makes the clip term a trust region
    # around the init rather than full PPO (where the old policy is
    # refreshed to the sampling policy each round), and the k1 KL
    # estimate below can go negative on off-policy samples.
    policy_lp, valid = _token_log_probs(policy_logits, labels, ignore_index)
    ref_lp, _ = _token_log_probs(reference_logits, labels, ignore_index)

    token_mask = valid * response_mask.to(torch.float32)
    lengths = token_mask.sum(dim=-1).clamp_min(1.0)
    policy_seq_lp = (policy_lp * token_mask).sum(dim=-1) / lengths
    ref_seq_lp = (ref_lp * token_mask).sum(dim=-1) / lengths

    advantages = rewards.to(torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

    ratios = torch.exp(policy_seq_lp - ref_seq_lp)
    clipped = torch.clamp(ratios, min=(1.0 - float(clip_epsilon)), max=(1.0 + float(clip_epsilon)))
    ppo_objective = torch.minimum(ratios * advantages, clipped * advantages)

    approx_kl = (policy_seq_lp - ref_seq_lp)
    return -(ppo_objective - float(kl_coefficient) * approx_kl).mean()


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _run_epoch(
    *,
    policy: ToyPolicyLM,
    reference: ToyPolicyLM,
    reward_model: ToyTokenRewardModel,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    clip_epsilon: float,
    kl_coefficient: float,
    ignore_index: int,
    max_batches: int | None,
) -> float:
    is_train = optimizer is not None
    policy.train(mode=is_train)
    reference.eval()
    reward_model.eval()

    total_loss = 0.0
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

        policy_logits = policy(model_inputs)
        with torch.no_grad():
            reference_logits = reference(model_inputs)
            rewards = reward_model(batch["input_ids"], batch["response_mask"])

        loss = ppo_policy_loss(
            policy_logits=policy_logits,
            reference_logits=reference_logits,
            labels=batch["labels"],
            response_mask=batch["response_mask"],
            rewards=rewards,
            clip_epsilon=float(clip_epsilon),
            kl_coefficient=float(kl_coefficient),
            ignore_index=int(ignore_index),
        )
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().item())
        steps += 1

    if steps == 0:
        return 0.0
    return total_loss / steps


def _write_samples(
    *,
    policy: ToyPolicyLM,
    reference: ToyPolicyLM,
    reward_model: ToyTokenRewardModel,
    loader,
    device: torch.device,
    out_path,
    epoch: int,
    clip_epsilon: float,
    kl_coefficient: float,
    ignore_index: int,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    batch = _move_batch(batch, device)
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    with torch.no_grad():
        policy_logits = policy(model_inputs)
        ref_logits = reference(model_inputs)
        rewards = reward_model(batch["input_ids"], batch["response_mask"])
        loss = ppo_policy_loss(
            policy_logits=policy_logits,
            reference_logits=ref_logits,
            labels=batch["labels"],
            response_mask=batch["response_mask"],
            rewards=rewards,
            clip_epsilon=float(clip_epsilon),
            kl_coefficient=float(kl_coefficient),
            ignore_index=int(ignore_index),
        )

    row = {
        "epoch": int(epoch),
        "input_ids": batch["input_ids"][0].detach().cpu().tolist(),
        "labels": batch["labels"][0].detach().cpu().tolist(),
        "reward": float(rewards[0].item()),
        "loss": float(loss.item()),
    }
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="llm",
        lesson="lesson_09_toy_rlhf_ppo",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("llm.toy_rlhf_ppo", log_file=paths.logs_dir / "train.log")
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
    policy = ToyPolicyLM(model_cfg).to(device_info.torch_device)
    reference = ToyPolicyLM(model_cfg).to(device_info.torch_device)
    reference.load_state_dict(policy.state_dict())
    for param in reference.parameters():
        param.requires_grad_(False)
    reference.eval()

    reward_model = ToyTokenRewardModel(
        pad_id=vocab.pad_id,
        good_token_id=vocab.good_token_id,
        bad_token_id=vocab.bad_token_id,
    ).to(device_info.torch_device)
    reward_model.eval()

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
        train_loss = _run_epoch(
            policy=policy,
            reference=reference,
            reward_model=reward_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device_info.torch_device,
            clip_epsilon=float(train_cfg.clip_epsilon),
            kl_coefficient=float(train_cfg.kl_coefficient),
            ignore_index=int(vocab.ignore_index),
            max_batches=train_cfg.max_train_batches,
        )
        with torch.no_grad():
            eval_loss = _run_epoch(
                policy=policy,
                reference=reference,
                reward_model=reward_model,
                loader=val_loader,
                optimizer=None,
                device=device_info.torch_device,
                clip_epsilon=float(train_cfg.clip_epsilon),
                kl_coefficient=float(train_cfg.kl_coefficient),
                ignore_index=int(vocab.ignore_index),
                max_batches=train_cfg.max_eval_batches,
            )

        _write_samples(
            policy=policy,
            reference=reference,
            reward_model=reward_model,
            loader=val_loader,
            device=device_info.torch_device,
            out_path=samples_path,
            epoch=epoch,
            clip_epsilon=float(train_cfg.clip_epsilon),
            kl_coefficient=float(train_cfg.kl_coefficient),
            ignore_index=int(vocab.ignore_index),
        )
        logger.info(
            "Epoch %d/%d | train loss %.4f | eval loss %.4f",
            epoch,
            train_cfg.epochs,
            train_loss,
            eval_loss,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=policy,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "llm", "lesson": "lesson_09_toy_rlhf_ppo", "vocab_size": vocab.size},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.llm.lesson_09_toy_rlhf_ppo.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
