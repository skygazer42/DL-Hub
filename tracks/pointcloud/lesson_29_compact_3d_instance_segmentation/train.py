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
from .model import ModelConfig, CompactInstanceSegmentation3DNet, instance_segmentation_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 29 (PointCloud): compact 3D instance segmentation."
    )

    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--num-points", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cluster-offset", type=float, default=0.9)
    parser.add_argument("--cluster-std", type=float, default=0.08)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        num_points=args.num_points,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        cluster_offset=args.cluster_offset,
        cluster_std=args.cluster_std,
    )
    model_cfg = ModelConfig(
        hidden_features=args.hidden_features,
        embedding_dim=args.embedding_dim,
        num_instances=2,
        dropout=args.dropout,
    )
    return train_cfg, data_cfg, model_cfg


def _run_epoch(
    *,
    model: CompactInstanceSegmentation3DNet,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss = 0.0
    total_acc = 0.0

    for step, (points, instance_ids) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        points = points.to(device)
        instance_ids = instance_ids.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            output = model(points)
            loss, parts = instance_segmentation_loss(output["logits"], instance_ids)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(points)
                loss, parts = instance_segmentation_loss(output["logits"], instance_ids)

        batch_size = int(points.size(0))
        total_examples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_acc += float(parts["point_acc"]) * batch_size

    if total_examples == 0:
        return {"loss": 0.0, "acc": 0.0}
    return {"loss": total_loss / total_examples, "acc": total_acc / total_examples}


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="pointcloud",
        lesson="lesson_29_compact_3d_instance_segmentation",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("pointcloud.instance_seg3d", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = CompactInstanceSegmentation3DNet(model_cfg).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "model": dataclass_to_dict(model_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            max_batches=train_cfg.max_train_batches,
        )
        eval_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            max_batches=train_cfg.max_eval_batches,
        )

        logger.info(
            "Epoch %d/%d | train loss %.6f | train acc %.4f | eval loss %.6f | eval acc %.4f",
            epoch,
            train_cfg.epochs,
            train_metrics["loss"],
            train_metrics["acc"],
            eval_metrics["loss"],
            eval_metrics["acc"],
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "eval_loss": eval_metrics["loss"],
                "eval_acc": eval_metrics["acc"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "pointcloud", "lesson": "lesson_29_compact_3d_instance_segmentation"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.pointcloud.lesson_29_compact_3d_instance_segmentation.train"
        )

    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
