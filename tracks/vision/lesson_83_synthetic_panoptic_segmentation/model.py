from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.panoptic_segmentation import build_panoptic_fpn_panoptic_segmenter


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    num_thing_classes: int = 3
    num_stuff_classes: int = 2
    max_instances: int = 2
    family: str = "panoptic_fpn"
    variant: str = "panoptic_fpn_tiny"
    width_mult: float = 0.5


class PanopticSegmentationModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if str(cfg.family).lower().strip() != "panoptic_fpn":
            raise ValueError("Only family='panoptic_fpn' is supported in this lesson.")
        if int(cfg.max_instances) <= 0:
            raise ValueError("max_instances must be > 0")
        self.cfg = cfg
        self.max_instances = int(cfg.max_instances)
        self.net = build_panoptic_fpn_panoptic_segmenter(
            in_channels=int(cfg.in_channels),
            num_thing_classes=int(cfg.num_thing_classes),
            num_stuff_classes=int(cfg.num_stuff_classes),
            variant=str(cfg.variant),
            width_mult=float(cfg.width_mult),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(images.to(torch.float32))


def panoptic_segmentation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    max_instances: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    semantic_labels = targets["semantic_labels"].to(torch.long)
    semantic_loss = F.cross_entropy(outputs["semantic_logits"], semantic_labels)

    k = int(max_instances)
    pred_cls = outputs["query_cls_logits"][:, :k, :]
    target_cls = targets["instance_classes"][:, :k].to(torch.long)
    instance_cls_loss = F.cross_entropy(pred_cls.reshape(-1, pred_cls.shape[-1]), target_cls.reshape(-1))

    pred_masks = outputs["mask_logits"][:, :k, :, :]
    target_masks = targets["instance_masks"][:, :k, :, :].to(torch.float32)
    instance_mask_loss = F.binary_cross_entropy_with_logits(pred_masks, target_masks)

    total = semantic_loss + 0.5 * instance_cls_loss + instance_mask_loss
    return total, {
        "semantic_loss": float(semantic_loss.detach().item()),
        "instance_cls_loss": float(instance_cls_loss.detach().item()),
        "instance_mask_loss": float(instance_mask_loss.detach().item()),
    }


def semantic_pixel_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels.to(torch.long)).to(torch.float32).mean().item())


__all__ = [
    "ModelConfig",
    "PanopticSegmentationModel",
    "panoptic_segmentation_loss",
    "semantic_pixel_accuracy",
]

