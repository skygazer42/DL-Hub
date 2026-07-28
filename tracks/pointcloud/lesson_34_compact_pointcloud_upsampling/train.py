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
from dlhub.pointcloud.ops import chamfer_distance
from dlhub.seed import set_seed

from .data import DataConfig, get_dataloaders
from .model import ModelConfig, build_model, list_supported_arches


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    arch: str = "punet_upsample:punet_upsample_tiny"
    width_mult: float = 1.0


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 34 (PointCloud): compact sparse-to-dense pointcloud upsampling."
    )
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--num-sparse-points", type=int, default=64)
    parser.add_argument("--upsample-factor", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--p-sphere", type=float, default=0.5)
    parser.add_argument("--sphere-surface-noise", type=float, default=0.0)
    parser.add_argument("--cube-surface-noise", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument(
        "--arch",
        type=str,
        default="punet_upsample:punet_upsample_tiny",
        help="Supported: <family>:<variant> (try --list-arch)",
    )
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--list-arch", action="store_true", help="Print supported architectures and exit.")
    args = parser.parse_args()

    if args.list_arch:
        print("\n".join(list_supported_arches()))
        raise SystemExit(0)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        arch=args.arch,
        width_mult=args.width_mult,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        num_sparse_points=args.num_sparse_points,
        upsample_factor=args.upsample_factor,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        p_sphere=args.p_sphere,
        sphere_surface_noise=args.sphere_surface_noise,
        cube_surface_noise=args.cube_surface_noise,
        shuffle_points=True,
    )
    return train_cfg, data_cfg


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> float:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total = 0
    total_loss = 0.0

    for step, (sparse, dense) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break
        sparse = sparse.to(device)
        dense = dense.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(sparse)
        loss = chamfer_distance(pred, dense)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(sparse.size(0))
        total += batch_size
        total_loss += float(loss.item()) * batch_size

    if total == 0:
        return 0.0
    return total_loss / total


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="pointcloud",
        lesson="lesson_34_compact_pointcloud_upsampling",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("pointcloud.upsampling", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Arch: %s", train_cfg.arch)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = build_model(
        ModelConfig(
            in_channels=3,
            arch=str(train_cfg.arch),
            variant="",
            width_mult=float(train_cfg.width_mult),
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

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))
    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_cd = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            max_batches=train_cfg.max_train_batches,
        )
        eval_cd = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info(
            "Epoch %d/%d | train CD %.6f | eval CD %.6f",
            epoch,
            train_cfg.epochs,
            train_cd,
            eval_cd,
        )
        append_jsonl(metrics_path, {"epoch": epoch, "train_chamfer": train_cd, "eval_chamfer": eval_cd})

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "pointcloud", "lesson": "lesson_34_compact_pointcloud_upsampling"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.pointcloud.lesson_34_compact_pointcloud_upsampling.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
