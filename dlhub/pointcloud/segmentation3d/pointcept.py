import torch
from torch import nn

from ._common import GridSpec3D, PointVoxelFusionSegBase

_VARIANTS: dict[str, dict[str, object]] = {
    "pointcept_tiny": {"width": 48, "d": 8, "h": 24, "w": 24},
    "pointcept_small": {"width": 64, "d": 8, "h": 32, "w": 32},
    "pointcept_base": {"width": 96, "d": 10, "h": 40, "w": 40},
}


class PointceptSeg(nn.Module):
    """Pointcept semantic segmentation (compact point-voxel fusion baseline)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        d: int,
        h: int,
        w: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = PointVoxelFusionSegBase(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            grid=GridSpec3D(d=int(d), h=int(h), w=int(w)),
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def build_pointcept_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointcept_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    return PointceptSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(int(cfg["width"]) * float(width_mult)),
        d=int(cfg["d"]),
        h=int(cfg["h"]),
        w=int(cfg["w"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointcept_segmenter3d(in_channels=3, num_classes=6, variant="pointcept_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
