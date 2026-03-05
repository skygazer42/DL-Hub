from __future__ import annotations

import torch
from torch import nn

from ._common import PointNet2SegBase


_VARIANTS: dict[str, dict[str, object]] = {
    "pointcnn_tiny": {"width": 32},
    "pointcnn_small": {"width": 48},
    "pointcnn_base": {"width": 64},
}


class PointCNNSeg(nn.Module):
    """PointCNN semantic segmentation (toy): PointNet++ hierarchy as a proxy for X-Conv."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = PointNet2SegBase(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def build_pointcnn_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointcnn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointCNNSeg(in_channels=int(in_channels), num_classes=int(num_classes), width=width, dropout=float(dropout))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointcnn_segmenter3d(in_channels=3, num_classes=6, variant="pointcnn_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

