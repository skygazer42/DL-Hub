from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    in_channels: int = 1
    plate_length: int = 6
    hidden_channels: int = 20
    num_blocks: int = 2
    dropout: float = 0.1


class LicensePlateRecognizer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_ch = int(cfg.in_channels)
        hidden = int(cfg.hidden_channels)
        blocks: list[nn.Module] = []
        for idx in range(int(cfg.num_blocks)):
            out_ch = hidden * (2**idx)
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
        self.readout = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, int(cfg.plate_length))),
            nn.Dropout(float(cfg.dropout)),
        )
        self.head = nn.Linear(in_ch, int(cfg.vocab_size))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images.to(torch.float32))
        pooled = self.readout(features).squeeze(2).transpose(1, 2)
        return self.head(pooled)


def plate_sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def plate_sequence_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        return float((pred == labels).all(dim=1).to(torch.float32).mean().item())


__all__ = [
    "LicensePlateRecognizer",
    "ModelConfig",
    "plate_sequence_accuracy",
    "plate_sequence_loss",
]
