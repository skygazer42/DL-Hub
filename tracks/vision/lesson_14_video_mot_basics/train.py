from __future__ import annotations

import argparse
import json
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

    box_weight: float = 2.0
    score_weight: float = 1.0
    cls_weight: float = 1.0


@dataclass(frozen=True)
class Stats:
    loss: float
    box_loss: float
    score_loss: float
    cls_loss: float
    presence_acc: float
    mean_iou: float


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 14 (Vision): synthetic video MOT basics with local MOT zoo."
    )

    # Data
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--min-box-size", type=int, default=6)
    parser.add_argument("--max-box-size", type=int, default=14)
    parser.add_argument("--max-speed", type=float, default=2.5)

    # Model
    parser.add_argument(
        "--arch",
        type=str,
        default="mot2d:sort_tiny",
        help="MOT local arch id. Example: mot2d:sort_tiny",
    )
    parser.add_argument(
        "--list-arch", action="store_true", help="Print supported MOT architectures and exit."
    )
    parser.add_argument(
        "--arch-family",
        type=str,
        default=None,
        help="Optional filter for --list-arch (example: --list-arch --arch-family sort).",
    )
    parser.add_argument(
        "--arch-match",
        type=str,
        default=None,
        help="Optional substring filter for --list-arch (example: --list-arch --arch-match tiny).",
    )
    parser.add_argument(
        "--list-arch-families",
        action="store_true",
        help="Print supported MOT arch families and exit.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved train/data/model config as JSON and exit.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=None,
        help="Optional max number of lines printed by --list-* flags.",
    )
    parser.add_argument(
        "--list-sort",
        type=str,
        default="none",
        choices=["none", "alpha"],
        help="Optional sorting for --list-* outputs: none (default) | alpha.",
    )
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--box-weight", type=float, default=2.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)

    args = parser.parse_args()

    any_list_flag = bool(args.list_arch or args.list_arch_families)
    if args.arch_family is not None and not args.list_arch:
        parser.error("--arch-family is only valid with --list-arch.")
    if args.arch_match is not None and not args.list_arch:
        parser.error("--arch-match is only valid with --list-arch.")
    if args.list_limit is not None and not any_list_flag:
        parser.error("--list-limit is only valid with --list-arch / --list-arch-families.")
    if str(args.list_sort).lower().strip() != "none" and not any_list_flag:
        parser.error("--list-sort is only valid with --list-arch / --list-arch-families.")

    def _arch_family(spec: str) -> str:
        name = str(spec).split(":", 1)[-1].strip().lower()
        if "_" not in name:
            return name
        return name.rsplit("_", 1)[0]

    if args.list_arch:
        from .model import list_supported_arches

        arches = list_supported_arches()
        if args.arch_family is not None:
            fam = str(args.arch_family).strip().lower()
            arches = [a for a in arches if _arch_family(a) == fam]
            if len(arches) == 0:
                parser.error(
                    f"Unknown arch family: {args.arch_family!r}. Use --list-arch-families."
                )
        if args.arch_match is not None:
            needle = str(args.arch_match).strip().lower()
            arches = [a for a in arches if needle in str(a).lower()]
            if len(arches) == 0:
                parser.error(
                    f"No arches matched: {args.arch_match!r}. Use --list-arch to see all options."
                )
        if str(args.list_sort).lower().strip() == "alpha":
            arches = sorted(arches, key=lambda s: str(s).lower())
        if args.list_limit is not None:
            n = int(args.list_limit)
            if n < 0:
                parser.error("--list-limit must be >= 0")
            arches = arches[:n]
        print("\n".join(arches))
        raise SystemExit(0)

    if args.list_arch_families:
        from .model import list_supported_arches

        fams = sorted({_arch_family(spec) for spec in list_supported_arches()})
        if str(args.list_sort).lower().strip() == "alpha":
            fams = sorted(fams, key=lambda s: str(s).lower())
        if args.list_limit is not None:
            n = int(args.list_limit)
            if n < 0:
                parser.error("--list-limit must be >= 0")
            fams = fams[:n]
        print("\n".join(fams))
        raise SystemExit(0)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        box_weight=args.box_weight,
        score_weight=args.score_weight,
        cls_weight=args.cls_weight,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        image_size=args.image_size,
        max_objects=args.max_objects,
        num_classes=args.num_classes,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        noise_std=args.noise_std,
        min_box_size=args.min_box_size,
        max_box_size=args.max_box_size,
        max_speed=args.max_speed,
    )
    model_cfg = ModelConfig(
        arch=args.arch,
        in_channels=3,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        image_size=args.image_size,
        width_mult=args.width_mult,
        dropout=args.dropout,
    )

    if args.print_config:
        payload = {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "model": dataclass_to_dict(model_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(0)

    return train_cfg, data_cfg, model_cfg


def _aligned_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """IoU for aligned boxes (..., 4), both in normalized xyxy."""

    ax1, ay1, ax2, ay2 = box_a.unbind(dim=-1)
    bx1, by1, bx2, by2 = box_b.unbind(dim=-1)

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
) -> Stats:
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_box = 0.0
    total_score = 0.0
    total_cls = 0.0
    total_presence_acc = 0.0
    total_iou = 0.0
    total = 0

    for step, (video, targets) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        video = video.to(device)
        box_target = targets["boxes"].to(device)
        label_target = targets["labels"].to(device)
        present_target = targets["present"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        out = model.track(video)
        max_objects = int(box_target.shape[1])
        if int(out["track_boxes"].shape[1]) < max_objects:
            raise ValueError(
                "Tracker num_tracks is smaller than data max_objects. "
                f"num_tracks={int(out['track_boxes'].shape[1])}, max_objects={max_objects}"
            )

        pred_boxes = out["track_boxes"][:, :max_objects, :]
        pred_scores = out["track_scores"][:, :max_objects]
        pred_cls = out["cls_logits"][:, :max_objects, :]

        present_sum = present_target.sum().clamp(min=1.0)
        box_err = torch.nn.functional.smooth_l1_loss(
            pred_boxes, box_target, reduction="none"
        ).mean(dim=-1)
        box_loss = (box_err * present_target).sum() / present_sum

        score_loss = torch.nn.functional.binary_cross_entropy(
            pred_scores.clamp(min=1e-6, max=1.0 - 1e-6),
            present_target,
        )

        pos_mask = present_target > 0.5
        cls_loss = torch.tensor(0.0, device=device)
        if bool(pos_mask.any()):
            cls_loss = torch.nn.functional.cross_entropy(
                pred_cls[pos_mask], label_target[pos_mask]
            )

        loss = (
            float(cfg.box_weight) * box_loss
            + float(cfg.score_weight) * score_loss
            + float(cfg.cls_weight) * cls_loss
        )

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            b = int(video.shape[0])
            total += b
            total_loss += float(loss.item()) * b
            total_box += float(box_loss.item()) * b
            total_score += float(score_loss.item()) * b
            total_cls += float(cls_loss.item()) * b

            pred_present = (pred_scores >= 0.5).to(present_target.dtype)
            total_presence_acc += float((pred_present == present_target).float().mean().item()) * b

            iou = _aligned_iou(pred_boxes, box_target)
            mean_iou = (iou * present_target).sum() / present_sum
            total_iou += float(mean_iou.item()) * b

    denom = max(1, total)
    return Stats(
        loss=total_loss / denom,
        box_loss=total_box / denom,
        score_loss=total_score / denom,
        cls_loss=total_cls / denom,
        presence_acc=total_presence_acc / denom,
        mean_iou=total_iou / denom,
    )


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="vision",
        lesson="lesson_14_video_mot_basics",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.video_mot_basics", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    if int(model_cfg.seq_len) != int(data_cfg.seq_len):
        raise ValueError(
            f"seq_len mismatch between model ({model_cfg.seq_len}) and data ({data_cfg.seq_len})"
        )
    if int(model_cfg.image_size) != int(data_cfg.image_size):
        raise ValueError(
            "image_size mismatch between model "
            f"({model_cfg.image_size}) and data ({data_cfg.image_size})"
        )

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
        )
        eval_stats = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            max_batches=train_cfg.max_eval_batches,
            cfg=train_cfg,
        )

        logger.info(
            "Epoch %d/%d | train loss %.4f (box %.4f score %.4f cls %.4f) "
            "presence_acc %.3f iou %.3f | eval loss %.4f presence_acc %.3f iou %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.box_loss,
            train_stats.score_loss,
            train_stats.cls_loss,
            train_stats.presence_acc,
            train_stats.mean_iou,
            eval_stats.loss,
            eval_stats.presence_acc,
            eval_stats.mean_iou,
        )

        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_box_loss": train_stats.box_loss,
                "train_score_loss": train_stats.score_loss,
                "train_cls_loss": train_stats.cls_loss,
                "train_presence_acc": train_stats.presence_acc,
                "train_mean_iou": train_stats.mean_iou,
                "eval_loss": eval_stats.loss,
                "eval_presence_acc": eval_stats.presence_acc,
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
            "lesson": "lesson_14_video_mot_basics",
            "arch": str(model_cfg.arch),
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_14_video_mot_basics.train"
        )

    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
