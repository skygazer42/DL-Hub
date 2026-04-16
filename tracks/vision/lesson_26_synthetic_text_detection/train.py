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
from .model import ModelConfig, TextDetectionModel, bbox_iou, text_detection_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


@dataclass(frozen=True)
class TrainStats:
    loss: float
    bbox_loss: float
    score_loss: float
    iou: float
    score_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(description="Lesson 26 (Vision): synthetic text detection.")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--empty-fraction", type=float, default=0.25)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=3)
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
        in_channels=args.in_channels,
        empty_fraction=args.empty_fraction,
    )
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    return train_cfg, data_cfg, model_cfg


def _run_epoch(
    *,
    model: TextDetectionModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
) -> TrainStats:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_bbox_loss = 0.0
    total_score_loss = 0.0
    total_iou = 0.0
    total_score_acc = 0.0
    total_batches = 0

    for step, (images, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break
        images = images.to(device)
        target_batch = {key: value.to(device) for key, value in targets.items()}
        outputs = model(images)
        loss, parts = text_detection_loss(outputs, target_batch)
        pred_score = (torch.sigmoid(outputs["score_logits"]) >= 0.5).to(torch.float32)
        score_acc = float((pred_score == target_batch["score"]).to(torch.float32).mean().item())
        iou = float(bbox_iou(outputs["bbox"].detach(), target_batch["bbox"].detach()).mean().item())

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_bbox_loss += float(parts["bbox_loss"])
        total_score_loss += float(parts["score_loss"])
        total_iou += iou
        total_score_acc += score_acc
        total_batches += 1

    if total_batches == 0:
        return TrainStats(loss=0.0, bbox_loss=0.0, score_loss=0.0, iou=0.0, score_acc=0.0)
    return TrainStats(
        loss=total_loss / total_batches,
        bbox_loss=total_bbox_loss / total_batches,
        score_loss=total_score_loss / total_batches,
        iou=total_iou / total_batches,
        score_acc=total_score_acc / total_batches,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_26_synthetic_text_detection",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.text_detection", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = TextDetectionModel(model_cfg).to(device_info.torch_device)
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
            "Epoch %d/%d | train loss %.4f score_acc %.3f iou %.3f | eval loss %.4f score_acc %.3f iou %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.score_acc,
            train_stats.iou,
            eval_stats.loss,
            eval_stats.score_acc,
            eval_stats.iou,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_bbox_loss": train_stats.bbox_loss,
                "train_score_loss": train_stats.score_loss,
                "train_iou": train_stats.iou,
                "train_score_acc": train_stats.score_acc,
                "eval_loss": eval_stats.loss,
                "eval_bbox_loss": eval_stats.bbox_loss,
                "eval_score_loss": eval_stats.score_loss,
                "eval_iou": eval_stats.iou,
                "eval_score_acc": eval_stats.score_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_26_synthetic_text_detection"},
    )
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_26_synthetic_text_detection.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
