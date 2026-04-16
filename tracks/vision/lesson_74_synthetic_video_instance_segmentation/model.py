from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 18
    num_blocks: int = 3
    num_instances: int = 2


class SegmentationBlock(nn.Module):
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


class VideoInstanceSegmentationModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        num_blocks = max(1, int(cfg.num_blocks))
        blocks: list[nn.Module] = [SegmentationBlock(int(cfg.in_channels), hidden)]
        for _ in range(num_blocks - 1):
            blocks.append(SegmentationBlock(hidden, hidden))
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden, int(cfg.num_instances), kernel_size=1)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        x = clip.to(torch.float32)
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(x.shape)}")
        batch, seq_len, channels, height, width = x.shape
        feat = self.encoder(x.view(batch * seq_len, channels, height, width))
        logits = self.head(feat).view(batch, seq_len, -1, height, width)
        return {"instance_logits": logits}


def video_instance_segmentation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["instance_logits"]
    masks = targets["instance_masks"].to(torch.float32)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks)
    return bce_loss, {"mask_bce_loss": float(bce_loss.item())}


__all__ = [
    "ModelConfig",
    "VideoInstanceSegmentationModel",
    "video_instance_segmentation_loss",
]
