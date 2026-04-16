from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    hidden_channels: int = 24
    num_blocks: int = 3


class VideoFrameInterpolationModel(nn.Module):
    """Predict a middle frame from two endpoint frames."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_channels = int(cfg.in_channels)
        hidden = int(cfg.hidden_channels)
        blocks = max(1, int(cfg.num_blocks))

        layers: list[nn.Module] = [
            nn.Conv2d(2 * in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks - 1):
            layers.extend(
                [
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                ]
            )
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, in_channels, kernel_size=1)

    def forward(self, endpoints: torch.Tensor) -> dict[str, torch.Tensor]:
        endpoints = endpoints.to(torch.float32)
        if endpoints.ndim != 5:
            raise ValueError(f"Expected endpoints as (B,2,C,H,W), got {tuple(endpoints.shape)}")
        if int(endpoints.shape[1]) != 2:
            raise ValueError(f"Expected exactly 2 endpoint frames, got {int(endpoints.shape[1])}")

        b, _, c, h, w = endpoints.shape
        x = endpoints.reshape(b, 2 * c, h, w)
        mid = torch.sigmoid(self.head(self.body(x)))
        return {"mid": mid}


def frame_interpolation_loss(
    outputs: dict[str, torch.Tensor], target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = torch.nn.functional.l1_loss(outputs["mid"], target.to(torch.float32))
    return l1, {"l1_loss": float(l1.item())}


def build_model(cfg: ModelConfig) -> VideoFrameInterpolationModel:
    return VideoFrameInterpolationModel(cfg)


__all__ = [
    "ModelConfig",
    "VideoFrameInterpolationModel",
    "build_model",
    "frame_interpolation_loss",
]
