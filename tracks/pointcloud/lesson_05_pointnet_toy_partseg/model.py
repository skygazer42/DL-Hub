from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    hidden_features: int = 64
    num_classes: int = 2
    dropout: float = 0.1


class PointNetPartSeg(nn.Module):
    """Minimal PointNet-style part segmentation.

    Input: points (B, N, 3)
    Output: logits (B, N, num_classes)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        c_in = int(cfg.in_channels)
        h = int(cfg.hidden_features)
        self.mlp = nn.Sequential(
            nn.Conv1d(c_in, h, kernel_size=1, bias=False),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Conv1d(h, h * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(h * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(h * 2, h * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(h * 4),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Sequential(
            nn.Conv1d(h * 8, h * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(h * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Conv1d(h * 2, h, kernel_size=1, bias=False),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Conv1d(h, int(cfg.num_classes), kernel_size=1, bias=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != int(self.cfg.in_channels):
            raise ValueError(f"Expected points shape (B, N, {self.cfg.in_channels}), got {tuple(points.shape)}")

        x = points.to(torch.float32).transpose(1, 2)  # (B, C, N)
        feat = self.mlp(x)  # (B, F, N)
        global_feat = torch.max(feat, dim=2, keepdim=True).values  # (B, F, 1)
        global_feat = global_feat.expand(-1, -1, feat.shape[2])  # (B, F, N)
        fused = torch.cat([feat, global_feat], dim=1)  # (B, 2F, N)
        logits = self.seg_head(fused)  # (B, num_classes, N)
        return logits.transpose(1, 2).contiguous()  # (B, N, num_classes)


__all__ = ["ModelConfig", "PointNetPartSeg"]

