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


class FusionModel(nn.Module):
    """Tiny all-in-focus fusion model over complementary focused input pairs."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_channels = int(cfg.in_channels)
        hidden = int(cfg.hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(int(cfg.num_blocks))])
        self.head = nn.Conv2d(hidden, in_channels, kernel_size=3, padding=1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        near = inputs["near_focus"].to(torch.float32)
        far = inputs["far_focus"].to(torch.float32)
        x = torch.cat([near, far], dim=1)
        feat = self.blocks(self.stem(x))
        return torch.sigmoid(self.head(feat))


def fusion_loss(
    fused: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction_loss = torch.nn.functional.l1_loss(fused, target.to(torch.float32))
    diff_x = fused[:, :, :, 1:] - fused[:, :, :, :-1]
    diff_y = fused[:, :, 1:, :] - fused[:, :, :-1, :]
    consistency_loss = diff_x.abs().mean() + diff_y.abs().mean()
    total_loss = reconstruction_loss + 0.05 * consistency_loss
    parts = {
        "reconstruction_loss": float(reconstruction_loss.item()),
        "consistency_loss": float(consistency_loss.item()),
    }
    return total_loss, parts


def build_model(cfg: ModelConfig) -> FusionModel:
    return FusionModel(cfg)


__all__ = ["FusionModel", "ModelConfig", "build_model", "fusion_loss"]
