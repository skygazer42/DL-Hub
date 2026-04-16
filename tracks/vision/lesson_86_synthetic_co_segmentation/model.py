from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.co_segmentation_zoo import build_local_model


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    num_classes: int = 2
    set_size: int = 3
    image_size: int = 32
    arch: str = "coseg:siamese_coseg_small"
    width_mult: float = 1.0
    dropout: float = 0.0


class CoSegmentationModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = build_local_model(
            str(cfg.arch),
            in_channels=int(cfg.in_channels),
            num_classes=int(cfg.num_classes),
            set_size=int(cfg.set_size),
            image_size=int(cfg.image_size),
            width_mult=float(cfg.width_mult),
            dropout=float(cfg.dropout),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(images.to(torch.float32))
        logits = outputs["logits"]
        pred_mask = logits.argmax(dim=2).to(torch.long)
        return {
            "logits": logits,
            "mask": pred_mask,
            "group_tokens": outputs["group_tokens"],
            "match_map": outputs["match_map"],
        }


def mask_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    pred = (pred_mask.to(torch.float32) > 0.5).to(torch.float32)
    truth = (target_mask.to(torch.float32) > 0.5).to(torch.float32)
    intersection = float((pred * truth).sum().item())
    union = float(((pred + truth) > 0.0).to(torch.float32).sum().item())
    if union == 0.0:
        return 1.0
    return intersection / union


def co_segmentation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"]
    class_index = targets["class_index"].to(torch.long)
    b, t, c, h, w = logits.shape
    ce = torch.nn.functional.cross_entropy(
        logits.view(b * t, c, h, w),
        class_index.view(b * t, h, w),
    )
    foreground = torch.softmax(logits, dim=2)[:, :, 1]
    mask = targets["mask"].to(torch.float32)
    intersection = (foreground * mask).sum()
    dice = (2.0 * intersection + 1.0) / (foreground.sum() + mask.sum() + 1.0)
    dice_loss = 1.0 - dice
    total = ce + dice_loss
    return total, {"cross_entropy": float(ce.item()), "dice_loss": float(dice_loss.item())}


def build_model(cfg: ModelConfig) -> CoSegmentationModel:
    return CoSegmentationModel(cfg)


__all__ = [
    "CoSegmentationModel",
    "ModelConfig",
    "build_model",
    "co_segmentation_loss",
    "mask_iou",
]
