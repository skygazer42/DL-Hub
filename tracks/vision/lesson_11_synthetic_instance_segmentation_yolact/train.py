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
from .model import ModelConfig, TinyYOLACT, mask_logits_from_proto


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"

    cls_pos_weight: float = 30.0
    reg_weight: float = 2.0
    mask_weight: float = 1.0

    arch: str = "yolact_tiny"
    width_mult: float = 1.0


@dataclass(frozen=True)
class Stats:
    loss: float
    cls_loss: float
    reg_loss: float
    mask_loss: float
    center_acc: float
    mask_iou: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 11 (Vision): synthetic instance segmentation (YOLACT-style, toy-first)."
    )

    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--min-rect", type=int, default=10)
    parser.add_argument("--max-rect", type=int, default=28)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--cls-pos-weight", type=float, default=30.0)
    parser.add_argument("--reg-weight", type=float, default=2.0)
    parser.add_argument("--mask-weight", type=float, default=1.0)

    parser.add_argument("--arch", type=str, default="yolact_tiny")
    parser.add_argument("--width-mult", type=float, default=1.0)

    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        cls_pos_weight=args.cls_pos_weight,
        reg_weight=args.reg_weight,
        mask_weight=args.mask_weight,
        arch=args.arch,
        width_mult=args.width_mult,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        stride=args.stride,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        noise_std=args.noise_std,
        min_rect=args.min_rect,
        max_rect=args.max_rect,
    )
    return train_cfg, data_cfg


def _mask_iou(
    pred_logits: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5
) -> torch.Tensor:
    probs = torch.sigmoid(pred_logits)
    pred = (probs > float(threshold)).to(torch.float32)
    target = target.to(torch.float32)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return intersection / union.clamp_min(1e-6)


def _run_epoch(
    *,
    model: TinyYOLACT,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    cfg: TrainConfig,
    data_cfg: DataConfig,
) -> Stats:
    is_train = optimizer is not None

    cls_criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(cfg.cls_pos_weight)], device=device)
    )
    reg_criterion = torch.nn.SmoothL1Loss(reduction="mean")
    mask_criterion = torch.nn.BCEWithLogitsLoss()

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_mask = 0.0
    total_center = 0.0
    total_iou = 0.0
    total = 0

    for step, (x, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        x = x.to(device)
        cls_target = targets["cls_target"].to(device)
        reg_target = targets["reg_target"].to(device)
        pos_mask = targets["pos_mask"].to(device)
        mask_target = targets["mask"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        out = model(x)
        cls_logits = out["cls_logits"]  # (B, 1, Gh, Gw)
        reg = out["bbox_deltas"]  # (B, 4, Gh, Gw)

        cls_loss = cls_criterion(cls_logits, cls_target)

        pred_pos = (reg * pos_mask).sum(dim=(2, 3))  # (B, 4)
        target_pos = (reg_target * pos_mask).sum(dim=(2, 3))  # (B, 4)
        reg_loss = reg_criterion(pred_pos, target_pos)

        mask_logits = mask_logits_from_proto(
            proto=out["proto"],
            mask_coeffs=out["mask_coeffs"],
            pos_mask=pos_mask,
            out_hw=(int(data_cfg.image_size), int(data_cfg.image_size)),
        )
        mask_loss = mask_criterion(mask_logits, mask_target)

        loss = cls_loss + float(cfg.reg_weight) * reg_loss + float(cfg.mask_weight) * mask_loss

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            b = int(x.shape[0])
            total += b
            total_loss += float(loss.item()) * b
            total_cls += float(cls_loss.item()) * b
            total_reg += float(reg_loss.item()) * b
            total_mask += float(mask_loss.item()) * b

            pred_idx = cls_logits.view(b, -1).argmax(dim=1)
            true_idx = cls_target.view(b, -1).argmax(dim=1)
            total_center += float((pred_idx == true_idx).float().mean().item()) * b

            total_iou += float(_mask_iou(mask_logits, mask_target).mean().item()) * b

    denom = max(1, total)
    return Stats(
        loss=total_loss / denom,
        cls_loss=total_cls / denom,
        reg_loss=total_reg / denom,
        mask_loss=total_mask / denom,
        center_acc=total_center / denom,
        mask_iou=total_iou / denom,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="vision",
        lesson="lesson_11_synthetic_instance_segmentation_yolact",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.instance_seg", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Arch: %s", train_cfg.arch)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = TinyYOLACT(
        ModelConfig(
            in_channels=1,
            num_classes=1,
            variant=str(train_cfg.arch),
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
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            max_batches=train_cfg.max_train_batches,
            cfg=train_cfg,
            data_cfg=data_cfg,
        )
        eval_stats = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            max_batches=train_cfg.max_eval_batches,
            cfg=train_cfg,
            data_cfg=data_cfg,
        )

        logger.info(
            "Epoch %d/%d | train loss %.4f (cls %.4f reg %.4f mask %.4f) | eval loss %.4f iou %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.cls_loss,
            train_stats.reg_loss,
            train_stats.mask_loss,
            eval_stats.loss,
            eval_stats.mask_iou,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_cls": train_stats.cls_loss,
                "train_reg": train_stats.reg_loss,
                "train_mask": train_stats.mask_loss,
                "train_center_acc": train_stats.center_acc,
                "train_mask_iou": train_stats.mask_iou,
                "eval_loss": eval_stats.loss,
                "eval_mask_iou": eval_stats.mask_iou,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_11_synthetic_instance_segmentation_yolact"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_11_synthetic_instance_segmentation_yolact.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
