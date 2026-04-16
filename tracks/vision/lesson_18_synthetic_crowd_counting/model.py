from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    depth: int = 4
    dropout: float = 0.0


class CrowdCountingRegressor(nn.Module):
    """Small fully-convolutional regressor that preserves spatial resolution."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        depth = int(cfg.depth)
        if hidden < 4:
            raise ValueError("hidden_channels must be >= 4")
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers: list[nn.Module] = []
        in_ch = int(cfg.in_channels)
        for _ in range(depth):
            layers.append(nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden))
            layers.append(nn.ReLU(inplace=True))
            if float(cfg.dropout) > 0.0:
                layers.append(nn.Dropout2d(float(cfg.dropout)))
            in_ch = hidden
        layers.append(nn.Conv2d(hidden, 1, kernel_size=1))
        self.features = nn.Sequential(*layers)
        self.output_activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        density = self.features(x)
        return self.output_activation(density)


__all__ = ["CrowdCountingRegressor", "ModelConfig"]
