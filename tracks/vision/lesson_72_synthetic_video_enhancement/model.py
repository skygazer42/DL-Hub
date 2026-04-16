from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 20
    num_blocks: int = 3


class ResidualFrameBlock(nn.Module):
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


class VideoEnhancementModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        num_blocks = max(1, int(cfg.num_blocks))
        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualFrameBlock(hidden) for _ in range(num_blocks)])
        self.head = nn.Conv2d(hidden, int(cfg.in_channels), kernel_size=1)

    def forward(self, degraded_clip: torch.Tensor) -> dict[str, torch.Tensor]:
        x = degraded_clip.to(torch.float32)
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(x.shape)}")
        batch, seq_len, channels, height, width = x.shape
        feat = self.stem(x.view(batch * seq_len, channels, height, width))
        feat = self.blocks(feat)
        residual = self.head(feat).view(batch, seq_len, channels, height, width)
        enhanced = (x + residual).clamp(0.0, 1.0)
        return {"enhanced": enhanced}


def video_enhancement_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = outputs["enhanced"]
    target = targets["clean"].to(torch.float32)
    reconstruction_loss = torch.nn.functional.mse_loss(pred, target)
    return reconstruction_loss, {"reconstruction_loss": float(reconstruction_loss.item())}


def psnr_from_mse(mse_value: float) -> float:
    return float(10.0 * torch.log10(torch.tensor(1.0 / max(float(mse_value), 1e-8))).item())


__all__ = ["ModelConfig", "VideoEnhancementModel", "psnr_from_mse", "video_enhancement_loss"]
