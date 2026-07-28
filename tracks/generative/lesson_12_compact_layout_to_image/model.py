from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 4
    hidden_channels: int = 32


class CompactLayoutToImageModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_channels = int(cfg.num_classes)
        hidden = int(cfg.hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, layout: torch.Tensor) -> torch.Tensor:
        return self.net(layout)

    def generate(self, layout: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(layout))
