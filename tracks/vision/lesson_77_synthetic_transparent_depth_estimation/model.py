from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    hidden_channels: int = 24
    num_blocks: int = 3


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class TransparentDepthEstimator(nn.Module):
    """Tiny dual-head model for depth and transparency estimation."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(int(cfg.num_blocks))])
        self.depth_head = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        self.transparency_head = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = image.to(torch.float32)
        feat = self.blocks(self.stem(image))
        depth = torch.sigmoid(self.depth_head(feat))
        transparency = torch.sigmoid(self.transparency_head(feat))
        return {"depth": depth, "transparency": transparency}


def build_model(cfg: ModelConfig) -> TransparentDepthEstimator:
    return TransparentDepthEstimator(cfg)


def transparent_depth_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    depth_loss = torch.nn.functional.smooth_l1_loss(
        outputs["depth"].to(torch.float32),
        targets["depth"].to(torch.float32),
    )
    mask_loss = torch.nn.functional.binary_cross_entropy(
        outputs["transparency"].to(torch.float32).clamp(1e-4, 1.0 - 1e-4),
        targets["transparency"].to(torch.float32),
    )
    total_loss = depth_loss + 0.2 * mask_loss
    return total_loss, {
        "depth_loss": float(depth_loss.detach().item()),
        "mask_loss": float(mask_loss.detach().item()),
    }


__all__ = [
    "ModelConfig",
    "TransparentDepthEstimator",
    "build_model",
    "transparent_depth_loss",
]
