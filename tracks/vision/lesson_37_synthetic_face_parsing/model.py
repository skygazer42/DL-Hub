from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FaceParsingConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_classes: int = 6


class FaceParsingSegmenter(nn.Module):
    def __init__(self, cfg: FaceParsingConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, int(cfg.num_classes), kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images.to(torch.float32))


def parsing_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, targets.to(torch.long))


def mean_iou(logits: torch.Tensor, targets: torch.Tensor, *, num_classes: int) -> float:
    pred = logits.argmax(dim=1)
    scores: list[float] = []
    for class_idx in range(int(num_classes)):
        pred_mask = pred == class_idx
        target_mask = targets == class_idx
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        intersection = (pred_mask & target_mask).sum().item()
        scores.append(float(intersection) / float(union))
    if not scores:
        return 1.0
    return float(sum(scores) / len(scores))


__all__ = ["FaceParsingConfig", "FaceParsingSegmenter", "mean_iou", "parsing_loss"]
