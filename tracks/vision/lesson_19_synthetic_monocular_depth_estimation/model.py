from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_blocks: int = 3


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DepthRegressor(nn.Module):
    """A tiny encoder-decoder that predicts a dense depth map from a monocular image."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        widths = [int(cfg.hidden_channels) * (2**i) for i in range(int(cfg.num_blocks))]

        self.stem = ConvBlock(int(cfg.in_channels), widths[0])
        self.encoder = nn.ModuleList(
            [ConvBlock(widths[i], widths[min(i + 1, len(widths) - 1)]) for i in range(len(widths) - 1)]
        )
        self.decoder = nn.ModuleList(
            [ConvBlock(widths[i + 1] + widths[i], widths[i]) for i in range(len(widths) - 2, -1, -1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.head = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        skips: list[torch.Tensor] = []

        feat = self.stem(x)
        skips.append(feat)
        for block in self.encoder:
            feat = self.pool(feat)
            feat = block(feat)
            skips.append(feat)

        feat = skips.pop()
        for block in self.decoder:
            skip = skips.pop()
            feat = torch.nn.functional.interpolate(feat, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            feat = block(torch.cat([feat, skip], dim=1))

        return torch.sigmoid(self.head(feat))


def build_model(cfg: ModelConfig) -> DepthRegressor:
    return DepthRegressor(cfg)


__all__ = ["DepthRegressor", "ModelConfig", "build_model"]

