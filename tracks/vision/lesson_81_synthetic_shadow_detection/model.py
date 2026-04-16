from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.shadow_detection.context_shadow import (
    build_context_shadow_shadow_detector,
)


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    hidden_channels: int = 24
    backbone_variant: str = "context_shadow_small"
    backbone_width_mult: float = 1.0


class ShadowDetectionModel(nn.Module):
    """Tiny shadow detector with an auxiliary relighting head."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = build_context_shadow_shadow_detector(
            in_channels=int(cfg.in_channels),
            variant=str(cfg.backbone_variant),
            width_mult=float(cfg.backbone_width_mult),
        )
        hidden = int(cfg.hidden_channels)
        self.relight_head = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels) + 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, int(cfg.in_channels), kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = image.to(torch.float32)
        shadow_outputs = self.backbone(image)
        shadow_mask = shadow_outputs["mask"]
        boundary = shadow_outputs["boundary"]
        relight_in = torch.cat([image, shadow_mask, boundary], dim=1)
        lit_image = torch.sigmoid(self.relight_head(relight_in))
        return {
            "logits": shadow_outputs["logits"],
            "shadow_mask": shadow_mask,
            "boundary": boundary,
            "lit_image": lit_image,
        }


def shadow_detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["logits"],
        targets["shadow_mask"].to(torch.float32),
    )
    boundary_loss = torch.nn.functional.binary_cross_entropy(
        outputs["boundary"],
        targets["boundary"].to(torch.float32),
    )
    lit_image_loss = torch.nn.functional.l1_loss(
        outputs["lit_image"],
        targets["lit_image"].to(torch.float32),
    )
    total_loss = mask_loss + 0.5 * boundary_loss + 0.8 * lit_image_loss
    parts = {
        "mask_loss": float(mask_loss.item()),
        "boundary_loss": float(boundary_loss.item()),
        "lit_image_loss": float(lit_image_loss.item()),
    }
    return total_loss, parts


def build_model(cfg: ModelConfig) -> ShadowDetectionModel:
    return ShadowDetectionModel(cfg)


__all__ = [
    "ModelConfig",
    "ShadowDetectionModel",
    "build_model",
    "shadow_detection_loss",
]
