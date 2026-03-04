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
from .model import ModelConfig, build_model, list_supported_arches


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"

    arch: str = "moco_pointnet:moco_pointnet_small"
    dropout: float = 0.0
    queue_size: int | None = None

    temperature: float = 0.2
    ema_decay: float = 0.99


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(description="Lesson 14 (PointCloud): self-supervised MoCo v2 (toy-first).")

    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--p-sphere", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument(
        "--arch",
        type=str,
        default="moco_pointnet:moco_pointnet_small",
        help="Supported: moco_pointnet:<variant> (try --list-arch)",
    )
    parser.add_argument("--list-arch", action="store_true", help="Print supported architectures and exit.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--queue-size", type=int, default=None, help="Override queue size (negatives).")

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.99)

    args = parser.parse_args()

    if args.list_arch:
        print("\n".join(list_supported_arches()))
        raise SystemExit(0)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        arch=args.arch,
        dropout=args.dropout,
        queue_size=args.queue_size,
        temperature=args.temperature,
        ema_decay=args.ema_decay,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        num_points=args.num_points,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        p_sphere=args.p_sphere,
    )
    return train_cfg, data_cfg


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    temperature: float,
    ema_decay: float,
) -> float:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total = 0
    total_loss = 0.0

    for step, (v1, v2, _y) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        v1 = v1.to(device)
        v2 = v2.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            model.momentum_update_key_encoder(ema_decay=float(ema_decay))

        out = model(v1, v2, temperature=float(temperature))
        loss = torch.nn.functional.cross_entropy(out["logits"], out["labels"])

        if is_train:
            loss.backward()
            optimizer.step()
            model.dequeue_and_enqueue(out["k"])

        bs = int(v1.size(0))
        total += bs
        total_loss += float(loss.item()) * bs

    if total == 0:
        return 0.0
    return total_loss / total


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="pointcloud",
        lesson="lesson_14_pointcloud_selfsupervised_moco",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("pointcloud.ssl_moco", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Arch: %s", train_cfg.arch)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)

    model = build_model(
        ModelConfig(
            arch=str(train_cfg.arch),
            variant="",
            in_channels=3,
            dropout=float(train_cfg.dropout),
            queue_size=train_cfg.queue_size,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            max_batches=train_cfg.max_train_batches,
            temperature=float(train_cfg.temperature),
            ema_decay=float(train_cfg.ema_decay),
        )
        eval_loss = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            max_batches=train_cfg.max_eval_batches,
            temperature=float(train_cfg.temperature),
            ema_decay=float(train_cfg.ema_decay),
        )

        logger.info("Epoch %d/%d | train loss %.4f | eval loss %.4f", epoch, train_cfg.epochs, train_loss, eval_loss)
        append_jsonl(metrics_path, {"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss})

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "pointcloud", "lesson": "lesson_14_pointcloud_selfsupervised_moco"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.pointcloud.lesson_14_pointcloud_selfsupervised_moco.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())

