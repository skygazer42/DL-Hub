from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 2
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


class VideoMattingModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(int(cfg.num_blocks))])
        self.head = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)

    def forward(self, video: torch.Tensor, trimap: torch.Tensor) -> torch.Tensor:
        video = video.to(torch.float32)
        trimap = trimap.to(torch.float32)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape (B,T,C,H,W), got {tuple(video.shape)}")
        if trimap.shape != video.shape:
            raise ValueError(
                f"Expected trimap shape to match video shape, got {tuple(trimap.shape)} vs {tuple(video.shape)}"
            )

        b, t, c, h, w = video.shape
        fused = torch.cat((video, trimap), dim=2).view(b * t, c * 2, h, w)
        features = self.blocks(self.stem(fused))
        logits = self.head(features)
        return logits.view(b, t, 1, h, w)


def video_matting_mae(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.sigmoid(logits).to(torch.float32)
    truth = target.to(torch.float32)
    return float(torch.mean(torch.abs(pred - truth)).item())


def video_matting_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    truth = target.to(torch.float32)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth)
    l1_loss = torch.mean(torch.abs(torch.sigmoid(logits) - truth))
    total = bce_loss + l1_loss
    return total, {"bce_loss": float(bce_loss.item()), "l1_loss": float(l1_loss.item())}


__all__ = [
    "ModelConfig",
    "VideoMattingModel",
    "video_matting_loss",
    "video_matting_mae",
]

