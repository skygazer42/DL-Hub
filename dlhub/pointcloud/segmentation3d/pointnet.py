from __future__ import annotations

import torch
from torch import nn

from ._common import PointNetSegBase


_VARIANTS: dict[str, dict[str, object]] = {
    "pointnet_tiny": {"width": 32, "depth": 2},
    "pointnet_small": {"width": 64, "depth": 3},
    "pointnet_base": {"width": 96, "depth": 4},
}


class PointNetSeg(nn.Module):
    """PointNet semantic segmentation (toy): point MLP + global context."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = PointNetSegBase(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def build_pointnet_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointNetSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointnet_segmenter3d(in_channels=3, num_classes=6, variant="pointnet_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    (y.mean()).backward()
    print("logits:", tuple(y.shape))

