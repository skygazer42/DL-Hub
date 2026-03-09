
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 32
    num_blocks: int = 3
    dropout: float = 0.0


class KeypointRegressor(nn.Module):
    """A small CNN that predicts `(x_norm, y_norm)` in [0, 1]."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_ch = int(cfg.in_channels)
        hidden = int(cfg.hidden_channels)
        blocks: list[nn.Module] = []

        for i in range(int(cfg.num_blocks)):
            out_ch = hidden * (2**i)
            blocks.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(in_ch, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        feats = self.backbone(x)
        out = self.head(feats)
        return torch.sigmoid(out)


__all__ = ["KeypointRegressor", "ModelConfig"]

