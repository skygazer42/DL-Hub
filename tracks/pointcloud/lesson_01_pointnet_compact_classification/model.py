from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_features: int = 64
    num_classes: int = 2
    dropout: float = 0.1


class PointNetClassifier(nn.Module):
    """A minimal PointNet-style classifier.

    Input: (B, N, 3)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        h = int(cfg.hidden_features)
        self.mlp = nn.Sequential(
            nn.Conv1d(3, h, kernel_size=1),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Conv1d(h, h * 2, kernel_size=1),
            nn.BatchNorm1d(h * 2),
            nn.ReLU(),
            nn.Conv1d(h * 2, h * 4, kernel_size=1),
            nn.BatchNorm1d(h * 4),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(h * 4, h * 2),
            nn.ReLU(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(h * 2, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points shape (B, N, 3), got {tuple(points.shape)}")
        x = points.transpose(1, 2)  # (B, 3, N)
        x = self.mlp(x)  # (B, C, N)
        x = torch.max(x, dim=2).values  # (B, C)
        return self.head(x)
