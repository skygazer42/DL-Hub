from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_blocks: int = 3
    num_attributes: int = 3
    dropout: float = 0.0


class FaceAttributeClassifier(nn.Module):
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
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(in_ch, int(cfg.num_attributes)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images.to(torch.float32))
        return self.head(features)


def attribute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.to(torch.float32))


def attribute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(torch.float32)
        exact_match = torch.all(preds == targets.to(torch.float32), dim=1)
        return float(exact_match.to(torch.float32).mean().item())


__all__ = [
    "FaceAttributeClassifier",
    "ModelConfig",
    "attribute_accuracy",
    "attribute_loss",
]
