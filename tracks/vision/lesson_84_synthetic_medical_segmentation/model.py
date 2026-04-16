from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.medical_segmentation.attention_unet import (
    build_attention_unet_medical_segmenter,
)
from dlhub.vision.medical_segmentation.mamba_unet import (
    build_mamba_unet_medical_segmenter,
)
from dlhub.vision.medical_segmentation.unet import build_unet_medical_segmenter
from dlhub.vision.medical_segmentation.unetpp import build_unetpp_medical_segmenter


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    num_classes: int = 3
    backbone_family: str = "unet"
    backbone_variant: str = "unet_tiny"
    backbone_width_mult: float = 1.0


_BUILDERS = {
    "unet": build_unet_medical_segmenter,
    "unetpp": build_unetpp_medical_segmenter,
    "attention_unet": build_attention_unet_medical_segmenter,
    "mamba_unet": build_mamba_unet_medical_segmenter,
}


class MedicalSegmentationModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        family = str(cfg.backbone_family)
        if family not in _BUILDERS:
            raise ValueError(
                f"Unsupported backbone_family={family!r}. "
                f"Supported: {sorted(_BUILDERS.keys())}"
            )
        self.backbone = _BUILDERS[family](
            in_channels=int(cfg.in_channels),
            num_classes=int(cfg.num_classes),
            variant=str(cfg.backbone_variant),
            width_mult=float(cfg.backbone_width_mult),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(images.to(torch.float32))


def medical_segmentation_loss(
    outputs: dict[str, torch.Tensor], targets: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = torch.nn.functional.cross_entropy(
        outputs["logits"], targets.to(torch.long)
    )
    return ce, {"cross_entropy": float(ce.item())}


def mean_dice(logits: torch.Tensor, targets: torch.Tensor, *, num_classes: int) -> float:
    pred = logits.argmax(dim=1)
    scores: list[float] = []
    for class_idx in range(int(num_classes)):
        pred_mask = pred == class_idx
        target_mask = targets == class_idx
        intersection = (pred_mask & target_mask).sum().item()
        denom = pred_mask.sum().item() + target_mask.sum().item()
        if denom == 0:
            continue
        scores.append(float(2.0 * intersection) / float(denom))
    if not scores:
        return 1.0
    return float(sum(scores) / len(scores))


def build_model(cfg: ModelConfig) -> MedicalSegmentationModel:
    return MedicalSegmentationModel(cfg)


__all__ = [
    "ModelConfig",
    "MedicalSegmentationModel",
    "build_model",
    "medical_segmentation_loss",
    "mean_dice",
]

