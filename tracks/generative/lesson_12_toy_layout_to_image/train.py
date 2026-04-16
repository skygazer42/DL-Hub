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
from .model import ModelConfig, ToyLayoutToImageModel


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 12 (Generative): toy layout-to-image with a simple encoder-decoder model."
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)

    parser.add_argument("--hidden-channels", type=int, default=32)

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
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=args.data_seed,
        val_fraction=args.val_fraction,
    )
    model_cfg = ModelConfig(
        num_classes=args.num_classes,
        hidden_channels=args.hidden_channels,
    )
    return train_cfg, data_cfg, model_cfg


def _evaluate(
    model: ToyLayoutToImageModel,
    loader: torch.utils.data.DataLoader,
    *,
    loss_fn: torch.nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for batch_idx, (layout, target) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            layout = layout.to(device)
            target = target.to(device)
            logits = model(layout)
            loss = loss_fn(logits, target)
            bsz = int(layout.size(0))
            total_loss += float(loss.item()) * bsz
            total_seen += bsz
    return total_loss / max(1, total_seen)


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="generative",
        lesson="lesson_12_toy_layout_to_image",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("generative.toy_layout_to_image", log_file=paths.logs_dir / "train.log")
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
    model = ToyLayoutToImageModel(model_cfg).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_idx, (layout, target) in enumerate(train_loader):
            if train_cfg.max_train_batches is not None and batch_idx >= train_cfg.max_train_batches:
                break
            layout = layout.to(device_info.torch_device)
            target = target.to(device_info.torch_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(layout)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            bsz = int(layout.size(0))
            total_loss += float(loss.item()) * bsz
            total_seen += bsz

        train_loss = total_loss / max(1, total_seen)
        val_loss = _evaluate(
            model,
            val_loader,
            loss_fn=loss_fn,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info(
            "Epoch %d/%d | train_bce %.4f | val_bce %.4f",
            epoch,
            train_cfg.epochs,
            train_loss,
            val_loss,
        )
        append_jsonl(metrics_path, {"epoch": epoch, "train_bce": train_loss, "val_bce": val_loss})

    sample_layout, sample_target = next(iter(val_loader))
    sample_layout = sample_layout[:16].to(device_info.torch_device)
    sample_target = sample_target[:16].to(device_info.torch_device)
    sample_pred = model.generate(sample_layout)
    torch.save(
        {
            "layout": sample_layout.cpu(),
            "target": sample_target.cpu(),
            "prediction": sample_pred.cpu(),
        },
        paths.run_dir / "samples.pt",
    )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=train_cfg.epochs,
        extra={"track": "generative", "lesson": "lesson_12_toy_layout_to_image"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.generative.lesson_12_toy_layout_to_image.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
