from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 16
    num_blocks: int = 3


class VideoStabilizationModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_channels = int(cfg.in_channels)
        hidden = max(8, int(cfg.hidden_channels))
        blocks = max(1, int(cfg.num_blocks))
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
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
        layers.append(nn.Conv2d(hidden, in_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, jittered_clip: torch.Tensor) -> dict[str, torch.Tensor]:
        x = jittered_clip.to(torch.float32)
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        stabilized = self.net(x.view(b * t, c, h, w)).view(b, t, c, h, w).clamp(0.0, 1.0)
        return {"stabilized": stabilized}


def video_stabilization_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.nn.functional.mse_loss(outputs["stabilized"], targets["stabilized"].to(torch.float32))
    return loss, {"reconstruction_loss": float(loss.item())}


__all__ = ["ModelConfig", "VideoStabilizationModel", "video_stabilization_loss"]
