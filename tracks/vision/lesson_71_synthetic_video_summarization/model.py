from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 16
    num_blocks: int = 3
    seq_len: int = 6


class FrameBlock(nn.Module):
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


class VideoSummarizationModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        num_blocks = max(1, int(cfg.num_blocks))

        blocks: list[nn.Module] = [FrameBlock(int(cfg.in_channels), hidden)]
        for _ in range(num_blocks - 1):
            blocks.append(FrameBlock(hidden, hidden))
        self.frame_encoder = nn.Sequential(*blocks)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        x = clip.to(torch.float32)
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(x.shape)}")
        batch, seq_len, channels, height, width = x.shape
        feat = self.frame_encoder(x.view(batch * seq_len, channels, height, width))
        tokens = feat.mean(dim=(2, 3)).view(batch, seq_len, -1)
        tokens = self.temporal(tokens.transpose(1, 2)).transpose(1, 2)
        return {"importance_logits": self.head(tokens).squeeze(-1)}


def video_summarization_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    importance_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["importance_logits"],
        targets["importance"].to(torch.float32),
    )
    return importance_loss, {"importance_loss": float(importance_loss.item())}


def frame_importance_mae(importance_logits: torch.Tensor, importance: torch.Tensor) -> float:
    pred = torch.sigmoid(importance_logits).to(torch.float32)
    target = importance.to(torch.float32)
    return float(torch.abs(pred - target).mean().item())


__all__ = [
    "ModelConfig",
    "VideoSummarizationModel",
    "frame_importance_mae",
    "video_summarization_loss",
]
