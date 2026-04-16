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


class ReflectionRemovalModel(nn.Module):
    """Tiny residual model that predicts transmission and reflection layers."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(int(cfg.num_blocks))])
        self.transmission_head = nn.Conv2d(hidden, int(cfg.in_channels), kernel_size=3, padding=1)
        self.reflection_head = nn.Conv2d(hidden, int(cfg.in_channels), kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        feat = self.blocks(self.stem(x))
        transmission = torch.sigmoid(self.transmission_head(feat))
        reflection = torch.sigmoid(self.reflection_head(feat))
        return {"transmission": transmission, "reflection": reflection}


def reflection_removal_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    transmission_loss = torch.nn.functional.l1_loss(
        outputs["transmission"],
        targets["transmission"].to(torch.float32),
    )
    reflection_loss = torch.nn.functional.l1_loss(
        outputs["reflection"],
        targets["reflection"].to(torch.float32),
    )
    total_loss = transmission_loss + 0.5 * reflection_loss
    parts = {
        "transmission_loss": float(transmission_loss.item()),
        "reflection_loss": float(reflection_loss.item()),
    }
    return total_loss, parts


def build_model(cfg: ModelConfig) -> ReflectionRemovalModel:
    return ReflectionRemovalModel(cfg)


__all__ = [
    "ModelConfig",
    "ReflectionRemovalModel",
    "build_model",
    "reflection_removal_loss",
]
