from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 32
    stride: int = 4


class TinyFCOS(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Output stride is fixed to 4 via two stride-2 convs.
        self.backbone = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, int(cfg.hidden_channels), kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(cfg.hidden_channels), int(cfg.hidden_channels), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Conv2d(int(cfg.hidden_channels), 1, kernel_size=1)
        self.reg_head = nn.Conv2d(int(cfg.hidden_channels), 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        cls_logits = self.cls_head(feats)  # (B, 1, H/4, W/4)
        reg = torch.relu(self.reg_head(feats))  # distances must be non-negative
        return {"cls_logits": cls_logits, "reg": reg}


__all__ = ["TinyFCOS", "ModelConfig"]

