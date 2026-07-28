import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from dlhub.checkpoint import save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed

from .data import DataConfig, get_dataloaders
from .model import (
    ConsistencyModel,
    ConsistencySchedule,
    ModelConfig,
    consistency_training_loss,
    sample_consistency,
    update_ema,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    ema_decay: float = 0.99
    num_sample_steps: int = 1


def _maybe_save_image_grid(images: torch.Tensor, path: str | Path) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return
    save_image(images, path, nrow=8)


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig, ConsistencySchedule]:
    parser = argparse.ArgumentParser(
        description="Lesson 05 (Generative): compact consistency model on MNIST-like images."
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--num-sample-steps", type=int, default=1)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-embed-dim", type=int, default=32)

    parser.add_argument("--num-discretization-steps", type=int, default=20)
    parser.add_argument("--sigma-min", type=float, default=0.02)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--sigma-data", type=float, default=0.5)

    parser.add_argument("--dataset", type=str, default="fake", choices=["fake", "mnist"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        ema_decay=args.ema_decay,
        num_sample_steps=args.num_sample_steps,
    )
    data_cfg = DataConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=args.data_seed,
        val_fraction=args.val_fraction,
    )
    model_cfg = ModelConfig(hidden_dim=args.hidden_dim, time_embed_dim=args.time_embed_dim)
    schedule = ConsistencySchedule(
        num_steps=args.num_discretization_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
    )
    return train_cfg, data_cfg, model_cfg, schedule


def _flatten_images(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim != 4 or batch.shape[1:] != (1, 28, 28):
        raise ValueError(f"Expected batch shape (B, 1, 28, 28), got {tuple(batch.shape)}")
    return batch.view(batch.size(0), -1)


def _evaluate(
    model: ConsistencyModel,
    target_model: ConsistencyModel,
    loader: torch.utils.data.DataLoader,
    schedule: ConsistencySchedule,
    *,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            x0 = _flatten_images(images)
            loss = consistency_training_loss(model, target_model, x0, schedule)

            bsz = int(images.size(0))
            total_loss += float(loss.item()) * bsz
            total_seen += bsz
    return total_loss / max(1, total_seen)


def run_training(
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    schedule: ConsistencySchedule | None = None,
) -> int:
    if schedule is None:
        schedule = ConsistencySchedule()

    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="generative",
        lesson="lesson_05_compact_consistency_model",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("generative.compact_consistency_model", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Dataset: %s", data_cfg.dataset)
    logger.info("Outputs: %s", paths.run_dir)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "model": dataclass_to_dict(model_cfg),
            "schedule": dataclass_to_dict(schedule),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = ConsistencyModel(model_cfg, schedule).to(device_info.torch_device)
    target_model = copy.deepcopy(model)
    for param in target_model.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for batch_idx, images in enumerate(train_loader):
            if train_cfg.max_train_batches is not None and batch_idx >= train_cfg.max_train_batches:
                break

            images = images.to(device_info.torch_device)
            x0 = _flatten_images(images)

            optimizer.zero_grad(set_to_none=True)
            loss = consistency_training_loss(model, target_model, x0, schedule)
            loss.backward()
            optimizer.step()
            update_ema(target_model, model, train_cfg.ema_decay)

            bsz = int(images.size(0))
            total_loss += float(loss.item()) * bsz
            total_seen += bsz

        train_loss = total_loss / max(1, total_seen)
        val_loss = _evaluate(
            model,
            target_model,
            val_loader,
            schedule,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )

        logger.info(
            "Epoch %d/%d | train_consistency_mse %.4f | val_consistency_mse %.4f",
            epoch,
            train_cfg.epochs,
            train_loss,
            val_loss,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_consistency_mse": train_loss,
                "val_consistency_mse": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

        samples = sample_consistency(
            target_model,
            schedule,
            num_samples=64,
            device=device_info.torch_device,
            num_steps=train_cfg.num_sample_steps,
        )
        torch.save({"samples": samples}, paths.run_dir / "samples.pt")
        _maybe_save_image_grid(samples, paths.run_dir / "samples.png")

    refine_frames = sample_consistency(
        target_model,
        schedule,
        num_samples=16,
        device=device_info.torch_device,
        num_steps=max(4, train_cfg.num_sample_steps),
        return_all=True,
    )
    torch.save({"frames": refine_frames}, paths.run_dir / "refine_grid.pt")
    if refine_frames.numel() > 0:
        _maybe_save_image_grid(refine_frames[-1], paths.run_dir / "refine_grid.png")

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=train_cfg.epochs,
        extra={
            "track": "generative",
            "lesson": "lesson_05_compact_consistency_model",
            # Sampling uses the EMA target network, so persist it too —
            # the online weights alone cannot reproduce samples.pt.
            "target_model_state": target_model.state_dict(),
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.generative.lesson_05_compact_consistency_model.train"
        )
    train_cfg, data_cfg, model_cfg, schedule = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg, schedule)


if __name__ == "__main__":
    raise SystemExit(main())
