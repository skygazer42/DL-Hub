from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import dlhub.vision.layout_generation as layout_model_zoo


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    hidden_channels: int = 24
    family: str = "layouttransformer"
    variant: str = "layouttransformer_tiny"
    width_mult: float = 1.0


class LayoutGenerationModel(nn.Module):
    """Tiny layout generation model with dlhub backbone and task heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        builder_name = f"build_{cfg.family}_layout_generator"
        if not hasattr(layout_model_zoo, builder_name):
            raise ValueError(f"unknown layout generation family: {cfg.family}")
        builder = getattr(layout_model_zoo, builder_name)
        self.backbone = builder(
            in_channels=int(cfg.in_channels),
            variant=str(cfg.variant),
            width_mult=float(cfg.width_mult),
        )

        hidden = int(cfg.hidden_channels)
        in_channels = int(cfg.in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layout_head = nn.Conv2d(hidden, in_channels, kernel_size=1)
        self.occupancy_head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        condition = condition.to(torch.float32)
        backbone_layout = self.backbone(condition)
        feat = self.stem(torch.cat([condition, backbone_layout], dim=1))
        layout = torch.sigmoid(self.layout_head(feat))
        occupancy = torch.sigmoid(self.occupancy_head(feat))
        residual = layout - condition
        return {"layout": layout, "occupancy": occupancy, "residual": residual}


def layout_generation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    layout_loss = torch.nn.functional.l1_loss(outputs["layout"], targets["layout"].to(torch.float32))
    occupancy_loss = torch.nn.functional.binary_cross_entropy(
        outputs["occupancy"],
        targets["occupancy"].to(torch.float32),
    )
    total = layout_loss + 0.5 * occupancy_loss
    return total, {
        "layout_loss": float(layout_loss.item()),
        "occupancy_loss": float(occupancy_loss.item()),
    }


def build_model(cfg: ModelConfig) -> LayoutGenerationModel:
    return LayoutGenerationModel(cfg)


__all__ = [
    "ModelConfig",
    "LayoutGenerationModel",
    "build_model",
    "layout_generation_loss",
]
