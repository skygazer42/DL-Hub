from __future__ import annotations

import torch
from torch import nn

from ._common import GridSpec3D, PointVoxelFusionSegBase


_VARIANTS: dict[str, dict[str, object]] = {
    "spvcnn_tiny": {"width": 48, "grid": (6, 24, 24)},
    "spvcnn_small": {"width": 64, "grid": (8, 32, 32)},
    "spvcnn_base": {"width": 96, "grid": (10, 40, 40)},
}


class SPVCNNSeg(nn.Module):
    """SPVCNN semantic segmentation (toy): point-voxel fusion backbone."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        grid: tuple[int, int, int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d, h, w = (int(x) for x in grid)
        self.net = PointVoxelFusionSegBase(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            grid=GridSpec3D(d=d, h=h, w=w),
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def build_spvcnn_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "spvcnn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return SPVCNNSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        grid=tuple(cfg["grid"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_spvcnn_segmenter3d(in_channels=3, num_classes=6, variant="spvcnn_tiny")
    x = torch.randn(2, 256, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

