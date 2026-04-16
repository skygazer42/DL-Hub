from __future__ import annotations

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
from .model import ModelConfig, ToyWorldModelsModel, reward_mae, world_models_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    family: str = "rssm_world"
    variant: str = "rssm_world_tiny"
    width_mult: float = 1.0


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 51 (Generative): toy world-model training."
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--family", type=str, default="rssm_world")
    parser.add_argument("--variant", type=str, default="rssm_world_tiny")
    parser.add_argument("--width-mult", type=float, default=1.0)

    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--context-dim", type=int, default=12)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        family=args.family,
        variant=args.variant,
        width_mult=args.width_mult,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        in_channels=args.in_channels,
        action_dim=args.action_dim,
        context_dim=args.context_dim,
        seed=args.data_seed,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        action_dim=args.action_dim,
        context_dim=args.context_dim,
        family=args.family,
        variant=args.variant,
        width_mult=args.width_mult,
    )
    return train_cfg, data_cfg, model_cfg


def _evaluate(
    model: ToyWorldModelsModel,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_reward_mae = 0.0
    total_seen = 0
    with torch.no_grad():
        for batch_idx, (obs, action, prompt, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            obs = obs.to(device)
            action = action.to(device)
            prompt = prompt.to(device)
            moved_targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(obs=obs, action=action, prompt=prompt)
            loss, _ = world_models_loss(outputs, moved_targets)
            bsz = int(obs.size(0))
            total_loss += float(loss.item()) * bsz
            total_reward_mae += reward_mae(outputs, moved_targets) * bsz
            total_seen += bsz
    denom = max(1, total_seen)
    return total_loss / denom, total_reward_mae / denom


def run_training(
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="generative",
        lesson="lesson_51_toy_world_models",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("generative.toy_world_models", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "model": dataclass_to_dict(model_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = ToyWorldModelsModel(model_cfg).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, int(train_cfg.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_reward_mae = 0.0
        total_seen = 0
        for batch_idx, (obs, action, prompt, targets) in enumerate(train_loader):
            if train_cfg.max_train_batches is not None and batch_idx >= train_cfg.max_train_batches:
                break
            obs = obs.to(device_info.torch_device)
            action = action.to(device_info.torch_device)
            prompt = prompt.to(device_info.torch_device)
            moved_targets = {k: v.to(device_info.torch_device) for k, v in targets.items()}

            optimizer.zero_grad(set_to_none=True)
            outputs = model(obs=obs, action=action, prompt=prompt)
            loss, _ = world_models_loss(outputs, moved_targets)
            loss.backward()
            optimizer.step()

            bsz = int(obs.size(0))
            total_loss += float(loss.item()) * bsz
            total_reward_mae += reward_mae(outputs, moved_targets) * bsz
            total_seen += bsz

        train_loss = total_loss / max(1, total_seen)
        train_reward_mae = total_reward_mae / max(1, total_seen)
        eval_loss, eval_reward_mae = _evaluate(
            model,
            val_loader,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )

        logger.info(
            "Epoch %d/%d | train_loss %.4f | train_reward_mae %.4f | eval_loss %.4f | eval_reward_mae %.4f",
            epoch,
            train_cfg.epochs,
            train_loss,
            train_reward_mae,
            eval_loss,
            eval_reward_mae,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_reward_mae": train_reward_mae,
                "eval_loss": eval_loss,
                "eval_reward_mae": eval_reward_mae,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    sample_obs, sample_action, sample_prompt, sample_targets = next(iter(val_loader))
    sample_obs = sample_obs[:8].to(device_info.torch_device)
    sample_action = sample_action[:8].to(device_info.torch_device)
    sample_prompt = sample_prompt[:8].to(device_info.torch_device)
    sample_outputs = model(obs=sample_obs, action=sample_action, prompt=sample_prompt)
    torch.save(
        {
            "obs": sample_obs.cpu(),
            "action": sample_action.cpu(),
            "prompt": sample_prompt.cpu(),
            "next_obs": sample_targets["next_obs"][:8],
            "pred_reconstruction": sample_outputs["reconstruction"].detach().cpu(),
            "pred_reward": sample_outputs["reward"].detach().cpu(),
            "pred_done": sample_outputs["done"].detach().cpu(),
        },
        paths.run_dir / "samples.pt",
    )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "generative", "lesson": "lesson_51_toy_world_models"},
    )
    return 0


def main() -> int:
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
