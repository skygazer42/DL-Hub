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
from .model import ModelConfig, build_model, road_scene_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


@dataclass(frozen=True)
class Stats:
    loss: float
    lane_loss: float
    object_loss: float
    scene_loss: float
    scene_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 22 (Vision): synthetic road scene understanding."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-lane-slots", type=int, default=3)
    parser.add_argument("--num-object-types", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--num-scene-classes", type=int, default=4)

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
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        num_lane_slots=args.num_lane_slots,
        num_object_types=args.num_object_types,
        noise_std=args.noise_std,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        num_lane_slots=args.num_lane_slots,
        num_object_types=args.num_object_types,
        num_scene_classes=args.num_scene_classes,
    )
    return train_cfg, data_cfg, model_cfg


def _scene_accuracy(scene_logits: torch.Tensor, scene_target: torch.Tensor) -> float:
    pred = scene_logits.argmax(dim=-1)
    return float((pred == scene_target).to(torch.float32).mean().item())


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
) -> Stats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_lane_loss = 0.0
    total_object_loss = 0.0
    total_scene_loss = 0.0
    total_scene_acc = 0.0
    total_batches = 0

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        target_batch = {key: value.to(device) for key, value in targets.items()}
        outputs = model(images)
        loss, parts = road_scene_loss(outputs, target_batch)
        scene_acc = _scene_accuracy(outputs["scene_logits"].detach(), target_batch["scene_label"])

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_lane_loss += float(parts["lane_loss"])
        total_object_loss += float(parts["object_loss"])
        total_scene_loss += float(parts["scene_loss"])
        total_scene_acc += scene_acc
        total_batches += 1

    if total_batches == 0:
        return Stats(loss=0.0, lane_loss=0.0, object_loss=0.0, scene_loss=0.0, scene_acc=0.0)
    return Stats(
        loss=total_loss / total_batches,
        lane_loss=total_lane_loss / total_batches,
        object_loss=total_object_loss / total_batches,
        scene_loss=total_scene_loss / total_batches,
        scene_acc=total_scene_acc / total_batches,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_22_synthetic_road_scene_understanding",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.road_scene", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = build_model(model_cfg).to(device_info.torch_device)
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
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device_info.torch_device,
            max_batches=train_cfg.max_train_batches,
        )
        with torch.no_grad():
            eval_stats = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device_info.torch_device,
                max_batches=train_cfg.max_eval_batches,
            )

        logger.info(
            "Epoch %d/%d | train loss %.6f | train scene acc %.3f | eval loss %.6f | eval scene acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.scene_acc,
            eval_stats.loss,
            eval_stats.scene_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_lane_loss": train_stats.lane_loss,
                "train_object_loss": train_stats.object_loss,
                "train_scene_loss": train_stats.scene_loss,
                "train_scene_acc": train_stats.scene_acc,
                "eval_loss": eval_stats.loss,
                "eval_lane_loss": eval_stats.lane_loss,
                "eval_object_loss": eval_stats.object_loss,
                "eval_scene_loss": eval_stats.scene_loss,
                "eval_scene_acc": eval_stats.scene_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_22_synthetic_road_scene_understanding"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_22_synthetic_road_scene_understanding.train"
        )

    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
