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
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class StitchingModel(nn.Module):
    """Tiny overlap-aware fusion network over two panorama-aligned partial views."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_channels = int(cfg.in_channels)
        hidden = int(cfg.hidden_channels)
        num_blocks = max(0, int(cfg.num_blocks))
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(num_blocks)])
        self.head = nn.Conv2d(hidden, in_channels, kernel_size=3, padding=1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        left_view = inputs["left_view"].to(torch.float32)
        right_view = inputs["right_view"].to(torch.float32)
        features = self.stem(torch.cat([left_view, right_view], dim=1))
        features = self.blocks(features)
        return torch.sigmoid(self.head(features))


def stitching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction_loss = torch.nn.functional.l1_loss(prediction, target.to(torch.float32))
    seam_loss = (
        (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]).abs().mean()
        + (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]).abs().mean()
    )
    total_loss = reconstruction_loss + 0.05 * seam_loss
    parts = {
        "reconstruction_loss": float(reconstruction_loss.item()),
        "seam_loss": float(seam_loss.item()),
    }
    return total_loss, parts


def build_model(cfg: ModelConfig) -> StitchingModel:
    return StitchingModel(cfg)


__all__ = ["ModelConfig", "StitchingModel", "build_model", "stitching_loss"]
