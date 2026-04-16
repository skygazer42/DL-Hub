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


class LaneDetectionModel(nn.Module):
    """A tiny encoder-decoder predicting lane heatmaps and x-coordinate offsets."""

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
        self.head = nn.Conv2d(widths[0], 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
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

        pred = torch.sigmoid(self.head(feat))
        return {"heatmap": pred[:, :1], "offset": pred[:, 1:2]}


def lane_detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    heatmap_loss = torch.nn.functional.mse_loss(outputs["heatmap"], targets["heatmap"])
    mask = targets["mask"]
    offset_error = torch.abs(outputs["offset"] - targets["offset"]) * mask
    offset_loss = offset_error.sum() / mask.sum().clamp_min(1.0)
    loss = heatmap_loss + offset_loss
    parts = {
        "heatmap_loss": float(heatmap_loss.item()),
        "offset_loss": float(offset_loss.item()),
    }
    return loss, parts


def build_model(cfg: ModelConfig) -> LaneDetectionModel:
    return LaneDetectionModel(cfg)


__all__ = ["LaneDetectionModel", "ModelConfig", "build_model", "lane_detection_loss"]
