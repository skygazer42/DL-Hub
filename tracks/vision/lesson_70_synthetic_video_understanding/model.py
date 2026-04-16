from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 16
    num_blocks: int = 3
    num_classes: int = 4


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VideoUnderstandingModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        num_blocks = max(1, int(cfg.num_blocks))

        blocks: list[nn.Module] = [Conv3dBlock(int(cfg.in_channels), hidden)]
        for _ in range(num_blocks - 1):
            blocks.append(Conv3dBlock(hidden, hidden))
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, int(cfg.num_classes))

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        x = clip.to(torch.float32)
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.encoder(x)
        pooled = feat.mean(dim=(2, 3, 4))
        return {"event_logits": self.head(pooled)}


def video_understanding_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    event_loss = torch.nn.functional.cross_entropy(
        outputs["event_logits"],
        targets["event_label"].to(torch.long),
    )
    return event_loss, {"event_loss": float(event_loss.item())}


def video_understanding_accuracy(event_logits: torch.Tensor, event_label: torch.Tensor) -> float:
    pred = event_logits.argmax(dim=-1)
    return float((pred == event_label.to(torch.long)).to(torch.float32).mean().item())


__all__ = [
    "ModelConfig",
    "VideoUnderstandingModel",
    "video_understanding_accuracy",
    "video_understanding_loss",
]
