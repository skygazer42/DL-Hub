from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_blocks: int = 3


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class VideoRestorationModel(nn.Module):
    """Per-frame restoration with temporal loss supervision in training."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_channels = int(cfg.in_channels)
        hidden_channels = int(cfg.hidden_channels)
        num_blocks = int(cfg.num_blocks)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(max(1, num_blocks))])
        self.head = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, degraded_clip: torch.Tensor) -> dict[str, torch.Tensor]:
        video = degraded_clip.to(torch.float32)
        if video.ndim != 5:
            raise ValueError(f"Expected input shape (B,T,C,H,W), got {tuple(video.shape)}")
        b, t, c, h, w = video.shape
        frames = video.view(b * t, c, h, w)
        feat = self.stem(frames)
        feat = self.body(feat)
        restored = self.head(feat).view(b, t, c, h, w).clamp(0.0, 1.0)
        return {"restored": restored}


def restoration_loss(
    restored: torch.Tensor,
    clean_target: torch.Tensor,
    temporal_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    l1_loss = F.l1_loss(restored, clean_target.to(torch.float32))
    if restored.shape[1] > 1:
        restored_delta = restored[:, 1:] - restored[:, :-1]
        target_delta = clean_target[:, 1:].to(torch.float32) - clean_target[:, :-1].to(torch.float32)
        temporal_loss = F.l1_loss(restored_delta, target_delta)
    else:
        temporal_loss = torch.zeros((), dtype=restored.dtype, device=restored.device)

    total_loss = l1_loss + float(temporal_weight) * temporal_loss
    parts = {"l1_loss": float(l1_loss.item()), "temporal_loss": float(temporal_loss.item())}
    return total_loss, parts


def build_model(cfg: ModelConfig) -> VideoRestorationModel:
    return VideoRestorationModel(cfg)


__all__ = ["ModelConfig", "VideoRestorationModel", "build_model", "restoration_loss"]
