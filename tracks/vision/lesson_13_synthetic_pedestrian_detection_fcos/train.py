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
from .model import ModelConfig, build_model


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
    centerness_weight: float = 1.0


@dataclass(frozen=True)
class Stats:
    loss: float
    cls_loss: float
    reg_loss: float
    centerness_loss: float
    center_acc: float
    mean_iou: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 13 (Vision): synthetic pedestrian detection (FCOS-style)."
    )

    # Data
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--min-box-w", type=int, default=6)
    parser.add_argument("--max-box-w", type=int, default=14)
    parser.add_argument("--min-box-h", type=int, default=18)
    parser.add_argument("--max-box-h", type=int, default=44)

    # Model
    parser.add_argument(
        "--arch",
        type=str,
        default="dldet:pedestrian_fcos",
        help="Detection local arch id (must be FCOS-style output). Default: dldet:pedestrian_fcos",
    )
    parser.add_argument("--width-mult", type=float, default=0.5)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--cls-pos-weight", type=float, default=30.0)
    parser.add_argument("--reg-weight", type=float, default=2.0)
    parser.add_argument("--centerness-weight", type=float, default=1.0)

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
        centerness_weight=args.centerness_weight,
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
        min_box_w=args.min_box_w,
        max_box_w=args.max_box_w,
        min_box_h=args.min_box_h,
        max_box_h=args.max_box_h,
    )
    model_cfg = ModelConfig(
        arch=args.arch,
        in_channels=3,
        num_classes=1,
        width_mult=args.width_mult,
    )
    return train_cfg, data_cfg, model_cfg


def _decode_boxes_from_grid(
    *,
    scores: torch.Tensor,
    reg: torch.Tensor,
    stride: int,
    image_size: int,
) -> torch.Tensor:
    """Decode 1 bbox per image by selecting the best-scoring grid cell.

    - scores: (B, 1, Gh, Gw) higher is better
    - reg: (B, 4, Gh, Gw) l/t/r/b distances

    Returns: (B, 4) in (x1, y1, x2, y2) pixel coordinates.
    """

    b, _, gh, gw = scores.shape
    best = scores.view(b, -1).argmax(dim=1)  # (B,)
    py = best // gw
    px = best % gw

    reg = reg.permute(0, 2, 3, 1).contiguous()  # (B, Gh, Gw, 4)
    idx = (py * gw + px).to(torch.long)
    reg_flat = reg.view(b, -1, 4)
    ltrb = reg_flat[torch.arange(b, device=reg.device), idx]  # (B, 4)

    cx = (px.to(torch.float32) + 0.5) * float(stride)
    cy = (py.to(torch.float32) + 0.5) * float(stride)

    x1 = cx - ltrb[:, 0]
    y1 = cy - ltrb[:, 1]
    x2 = cx + ltrb[:, 2]
    y2 = cy + ltrb[:, 3]
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    boxes = boxes.clamp(min=0.0, max=float(image_size))
    return boxes


def _iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """IoU for aligned boxes: both are (B, 4) in xyxy."""

    ax1, ay1, ax2, ay2 = box_a.unbind(dim=1)
    bx1, by1, bx2, by2 = box_b.unbind(dim=1)

    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)

    iw = (ix2 - ix1).clamp(min=0.0)
    ih = (iy2 - iy1).clamp(min=0.0)
    inter = iw * ih

    area_a = (ax2 - ax1).clamp(min=0.0) * (ay2 - ay1).clamp(min=0.0)
    area_b = (bx2 - bx1).clamp(min=0.0) * (by2 - by1).clamp(min=0.0)
    union = (area_a + area_b - inter).clamp(min=1e-12)
    return inter / union


def _run_epoch(
    *,
    model: torch.nn.Module,
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
    center_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_center = 0.0
    total_acc = 0.0
    total_iou = 0.0
    total = 0

    for step, (x, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        x = x.to(device)
        cls_target = targets["cls_target"].to(device)
        reg_target = targets["reg_target"].to(device)
        pos_mask = targets["pos_mask"].to(device)
        box_target = targets["box"].to(device)
        cent_target = targets.get("centerness_target")
        if cent_target is not None:
            cent_target = cent_target.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        out = model(x)
        cls_logits = out["cls_logits"]
        reg = out["reg"]

        cls_loss = cls_criterion(cls_logits, cls_target)

        # Regression only at positive locations.
        pred_pos = (reg * pos_mask).sum(dim=(2, 3))
        target_pos = (reg_target * pos_mask).sum(dim=(2, 3))
        reg_loss = reg_criterion(pred_pos, target_pos)

        centerness_loss = torch.tensor(0.0, device=device)
        cent_logits = out.get("centerness")
        if cent_logits is not None and cent_target is not None:
            pred_center = (cent_logits * pos_mask).sum(dim=(2, 3))
            target_center = (cent_target * pos_mask).sum(dim=(2, 3))
            centerness_loss = center_criterion(pred_center, target_center)

        loss = (
            cls_loss
            + float(cfg.reg_weight) * reg_loss
            + float(cfg.centerness_weight) * centerness_loss
        )

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            b = int(x.shape[0])
            total += b
            total_loss += float(loss.item()) * b
            total_cls += float(cls_loss.item()) * b
            total_reg += float(reg_loss.item()) * b
            total_center += float(centerness_loss.item()) * b

            scores = torch.sigmoid(cls_logits)
            if cent_logits is not None:
                scores = scores * torch.sigmoid(cent_logits)

            pred_idx = scores.view(b, -1).argmax(dim=1)
            true_idx = cls_target.view(b, -1).argmax(dim=1)
            total_acc += float((pred_idx == true_idx).float().mean().item()) * b

            pred_boxes = _decode_boxes_from_grid(
                scores=scores,
                reg=reg,
                stride=int(data_cfg.stride),
                image_size=int(data_cfg.image_size),
            )
            total_iou += float(_iou(pred_boxes, box_target).mean().item()) * b

    denom = max(1, total)
    return Stats(
        loss=total_loss / denom,
        cls_loss=total_cls / denom,
        reg_loss=total_reg / denom,
        centerness_loss=total_center / denom,
        center_acc=total_acc / denom,
        mean_iou=total_iou / denom,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="vision",
        lesson="lesson_13_synthetic_pedestrian_detection_fcos",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.synth_pedestrian_fcos", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    if int(data_cfg.stride) != 4:
        raise ValueError("This lesson currently assumes stride=4 to match FCOS-style models.")

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
            "Epoch %d/%d | train loss %.4f (cls %.4f reg %.4f center %.4f) acc %.3f iou %.3f | "
            "eval loss %.4f acc %.3f iou %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.cls_loss,
            train_stats.reg_loss,
            train_stats.centerness_loss,
            train_stats.center_acc,
            train_stats.mean_iou,
            eval_stats.loss,
            eval_stats.center_acc,
            eval_stats.mean_iou,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_cls_loss": train_stats.cls_loss,
                "train_reg_loss": train_stats.reg_loss,
                "train_centerness_loss": train_stats.centerness_loss,
                "train_center_acc": train_stats.center_acc,
                "train_mean_iou": train_stats.mean_iou,
                "eval_loss": eval_stats.loss,
                "eval_center_acc": eval_stats.center_acc,
                "eval_mean_iou": eval_stats.mean_iou,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={
            "track": "vision",
            "lesson": "lesson_13_synthetic_pedestrian_detection_fcos",
            "arch": str(model_cfg.arch),
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos.train"
        )

    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
