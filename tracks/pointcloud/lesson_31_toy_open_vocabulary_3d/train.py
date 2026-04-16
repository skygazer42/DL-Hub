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

from .data import DataConfig, ToyOpenVocabulary3DDataset, get_dataloaders
from .model import ModelConfig, ToyOpenVocabulary3DModel, mask_iou, open_vocabulary_3d_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    text_dim: int = 24
    point_dim: int = 48
    hidden_dim: int = 48


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 31 (PointCloud): toy open-vocabulary 3D recognition and grounding."
    )
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--num-points", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=8)
    parser.add_argument("--jitter-std", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--text-dim", type=int, default=24)
    parser.add_argument("--point-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=48)
    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        text_dim=args.text_dim,
        point_dim=args.point_dim,
        hidden_dim=args.hidden_dim,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        num_points=args.num_points,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        max_text_length=args.max_text_length,
        jitter_std=args.jitter_std,
    )
    return train_cfg, data_cfg


def _run_epoch(
    *,
    model: ToyOpenVocabulary3DModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_examples = 0
    total_loss = 0.0
    total_class_acc = 0.0
    total_mask_iou = 0.0
    for step, (points, query_ids, query_mask, class_targets, mask_targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        points = points.to(device)
        query_ids = query_ids.to(device)
        query_mask = query_mask.to(device)
        class_targets = class_targets.to(device)
        mask_targets = mask_targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if is_train:
            outputs = model(points, query_ids, query_mask)
            loss, _ = open_vocabulary_3d_loss(
                outputs["class_logits"], outputs["mask_logits"], class_targets, mask_targets
            )
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(points, query_ids, query_mask)
                loss, _ = open_vocabulary_3d_loss(
                    outputs["class_logits"], outputs["mask_logits"], class_targets, mask_targets
                )

        batch_size = int(points.size(0))
        class_acc = (
            outputs["class_logits"].argmax(dim=1).eq(class_targets).to(torch.float32).mean().item()
        )
        batch_mask_iou = mask_iou(outputs["mask_logits"].detach(), mask_targets.detach())

        total_examples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_class_acc += float(class_acc) * batch_size
        total_mask_iou += float(batch_mask_iou) * batch_size

    if total_examples == 0:
        return {"loss": 0.0, "class_acc": 0.0, "mask_iou": 0.0}
    denom = float(total_examples)
    return {
        "loss": total_loss / denom,
        "class_acc": total_class_acc / denom,
        "mask_iou": total_mask_iou / denom,
    }


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="pointcloud",
        lesson="lesson_31_toy_open_vocabulary_3d",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("pointcloud.open_vocabulary_3d", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)
    vocab_ds = ToyOpenVocabulary3DDataset(data_cfg)
    model_cfg = ModelConfig(
        vocab_size=vocab_ds.vocab_size,
        pad_id=vocab_ds.pad_id,
        text_dim=int(train_cfg.text_dim),
        point_dim=int(train_cfg.point_dim),
        hidden_dim=int(train_cfg.hidden_dim),
        num_classes=3,
    )
    model = ToyOpenVocabulary3DModel(model_cfg).to(device_info.torch_device)
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
            (
                "Epoch %d/%d | train loss %.6f | train class acc %.4f | train mask IoU %.4f | "
                "eval loss %.6f | eval class acc %.4f | eval mask IoU %.4f"
            ),
            epoch,
            train_cfg.epochs,
            train_metrics["loss"],
            train_metrics["class_acc"],
            train_metrics["mask_iou"],
            eval_metrics["loss"],
            eval_metrics["class_acc"],
            eval_metrics["mask_iou"],
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_class_acc": train_metrics["class_acc"],
                "train_mask_iou": train_metrics["mask_iou"],
                "eval_loss": eval_metrics["loss"],
                "eval_class_acc": eval_metrics["class_acc"],
                "eval_mask_iou": eval_metrics["mask_iou"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "pointcloud", "lesson": "lesson_31_toy_open_vocabulary_3d"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.pointcloud.lesson_31_toy_open_vocabulary_3d.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
