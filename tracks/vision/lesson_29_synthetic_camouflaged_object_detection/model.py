from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
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


class CamouflagedObjectDetectionModel(nn.Module):
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(images.to(torch.float32)))
        return self.head(features)


def mask_iou(logits: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) >= float(threshold)).to(torch.float32)
    truth = (target.to(torch.float32) >= 0.5).to(torch.float32)
    intersection = float((pred * truth).sum().item())
    union = float(((pred + truth) > 0).to(torch.float32).sum().item())
    if union == 0.0:
        return 1.0
    return intersection / union


def camouflaged_detection_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    target = target.to(torch.float32)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum()
    dice = (2.0 * intersection + 1.0) / (probs.sum() + target.sum() + 1.0)
    dice_loss = 1.0 - dice
    total_loss = bce_loss + dice_loss
    return total_loss, {"bce_loss": float(bce_loss.item()), "dice_loss": float(dice_loss.item())}


__all__ = [
    "CamouflagedObjectDetectionModel",
    "ModelConfig",
    "camouflaged_detection_loss",
    "mask_iou",
]

