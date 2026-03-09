
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
from .model import ModelConfig, TinyYOLOv1


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"

    obj_pos_weight: float = 20.0
    cls_pos_weight: float = 20.0
    box_weight: float = 2.0


@dataclass(frozen=True)
class Stats:
    loss: float
    obj_loss: float
    cls_loss: float
    box_loss: float
    center_acc: float
    mean_iou: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(description="Lesson 12 (Vision): synthetic detection (YOLOv1-style).")

    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--min-box-size", type=int, default=10)
    parser.add_argument("--max-box-size", type=int, default=28)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--obj-pos-weight", type=float, default=20.0)
    parser.add_argument("--cls-pos-weight", type=float, default=20.0)
    parser.add_argument("--box-weight", type=float, default=2.0)

    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        obj_pos_weight=args.obj_pos_weight,
        cls_pos_weight=args.cls_pos_weight,
        box_weight=args.box_weight,
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
        min_box_size=args.min_box_size,
        max_box_size=args.max_box_size,
    )
    return train_cfg, data_cfg


def _decode_box_from_grid(
    *,
    obj_logits: torch.Tensor,
    bbox: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Decode 1 bbox per image by selecting the best objectness cell.

    bbox is normalized cx,cy,w,h in [0,1].
    Returns (B, 4) in pixel (x1,y1,x2,y2).
    """

    b, _, gh, gw = obj_logits.shape
    scores = obj_logits.view(b, -1)
    best = scores.argmax(dim=1)  # (B,)
    py = best // gw
    px = best % gw

    bbox = bbox.permute(0, 2, 3, 1).contiguous()  # (B, Gh, Gw, 4)
    idx = (py * gw + px).to(torch.long)
    bbox_flat = bbox.view(b, -1, 4)
    cxywh = bbox_flat[torch.arange(b, device=bbox.device), idx]  # (B, 4)

    cx = cxywh[:, 0] * float(image_size)
    cy = cxywh[:, 1] * float(image_size)
    w = cxywh[:, 2] * float(image_size)
    h = cxywh[:, 3] * float(image_size)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)


def _iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)

    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)

    area_a = (ax2 - ax1).clamp(min=0.0) * (ay2 - ay1).clamp(min=0.0)
    area_b = (bx2 - bx1).clamp(min=0.0) * (by2 - by1).clamp(min=0.0)
    union = (area_a + area_b - inter).clamp(min=1e-12)
    return inter / union


def _run_epoch(
    *,
    model: TinyYOLOv1,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    cfg: TrainConfig,
    data_cfg: DataConfig,
) -> Stats:
    is_train = optimizer is not None

    obj_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(cfg.obj_pos_weight)], device=device))
    cls_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(cfg.cls_pos_weight)], device=device))
    box_criterion = torch.nn.SmoothL1Loss(reduction="mean")

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_obj = 0.0
    total_cls = 0.0
    total_box = 0.0
    total_center = 0.0
    total_iou = 0.0
    total = 0

    for step, (x, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        x = x.to(device)
        obj_target = targets["obj_target"].to(device)
        cls_target = targets["cls_target"].to(device)
        bbox_target = targets["bbox_target"].to(device)
        pos_mask = targets["pos_mask"].to(device)
        box_target = targets["box"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        out = model(x)
        obj_logits = out["obj_logits"]
        cls_logits = out["cls_logits"]
        bbox = out["bbox"]

        obj_loss = obj_criterion(obj_logits, obj_target)
        cls_loss = cls_criterion(cls_logits, cls_target)

        # Regression only at positive locations.
        pos = pos_mask  # (B, 1, Gh, Gw)
        pred_pos = (bbox * pos).sum(dim=(2, 3))  # (B, 4)
        target_pos = (bbox_target * pos).sum(dim=(2, 3))  # (B, 4)
        box_loss = box_criterion(pred_pos, target_pos)

        loss = obj_loss + cls_loss + float(cfg.box_weight) * box_loss

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            b = int(x.shape[0])
            total += b
            total_loss += float(loss.item()) * b
            total_obj += float(obj_loss.item()) * b
            total_cls += float(cls_loss.item()) * b
            total_box += float(box_loss.item()) * b

            # center acc: argmax cell matches target cell.
            pred_idx = obj_logits.view(b, -1).argmax(dim=1)
            true_idx = obj_target.view(b, -1).argmax(dim=1)
            total_center += float((pred_idx == true_idx).float().mean().item()) * b

            pred_boxes = _decode_box_from_grid(
                obj_logits=obj_logits, bbox=bbox, image_size=int(data_cfg.image_size)
            )
            total_iou += float(_iou(pred_boxes, box_target).mean().item()) * b

    denom = max(1, total)
    return Stats(
        loss=total_loss / denom,
        obj_loss=total_obj / denom,
        cls_loss=total_cls / denom,
        box_loss=total_box / denom,
        center_acc=total_center / denom,
        mean_iou=total_iou / denom,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="vision", lesson="lesson_12_synthetic_detection_yolo", run_name=train_cfg.run_name
    )
    logger = get_logger("vision.synth_yolo", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    if int(data_cfg.stride) != 4:
        raise ValueError("This lesson currently assumes stride=4 to match the model definition.")

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = TinyYOLOv1(ModelConfig(in_channels=1, hidden_channels=32, stride=int(data_cfg.stride))).to(
        device_info.torch_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
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

        append_jsonl(metrics_path, {"epoch": epoch, "train": dataclass_to_dict(train_stats), "eval": dataclass_to_dict(eval_stats)})
        logger.info(
            "Epoch %d | train=%.4f (iou=%.3f) eval=%.4f (iou=%.3f)",
            epoch,
            train_stats.loss,
            train_stats.mean_iou,
            eval_stats.loss,
            eval_stats.mean_iou,
        )

        save_checkpoint(
            paths.checkpoints_dir / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            extra={"train_cfg": dataclass_to_dict(train_cfg), "data_cfg": dataclass_to_dict(data_cfg)},
        )

    logger.info("Done. Run dir: %s", paths.run_dir)
    return 0


if __name__ == "__main__":
    cfg_train, cfg_data = parse_args()
    raise SystemExit(run_training(cfg_train, cfg_data))

