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


class DerainingModel(nn.Module):
    """Tiny residual deraining model with restoration and rain-layer heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(int(cfg.num_blocks))])
        self.restored_head = nn.Conv2d(hidden, int(cfg.in_channels), kernel_size=3, padding=1)
        self.rain_head = nn.Conv2d(hidden, int(cfg.in_channels), kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        feat = self.blocks(self.stem(x))
        restored = torch.sigmoid(self.restored_head(feat))
        rain_layer = torch.sigmoid(self.rain_head(feat))
        return {"restored": restored, "rain_layer": rain_layer}


def deraining_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction_loss = torch.nn.functional.l1_loss(
        outputs["restored"],
        targets["clean"].to(torch.float32),
    )
    rain_loss = torch.nn.functional.l1_loss(
        outputs["rain_layer"],
        targets["rain_layer"].to(torch.float32),
    )
    total_loss = reconstruction_loss + 0.5 * rain_loss
    parts = {
        "reconstruction_loss": float(reconstruction_loss.item()),
        "rain_loss": float(rain_loss.item()),
    }
    return total_loss, parts


def build_model(cfg: ModelConfig) -> DerainingModel:
    return DerainingModel(cfg)


__all__ = ["DerainingModel", "ModelConfig", "build_model", "deraining_loss"]
