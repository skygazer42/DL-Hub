import math

import torch
from torch import nn

from ._common import (
    GridSpec2D,
    PointMLP,
    TinyUNet2D,
    check_points,
    gather_2d,
    scatter_mean_2d,
    split_xyz_features,
)

_VARIANTS: dict[str, dict[str, object]] = {
    "polarnet_tiny": {"width": 48, "h": 32, "w": 64},
    "polarnet_small": {"width": 64, "h": 40, "w": 96},
    "polarnet_base": {"width": 96, "h": 48, "w": 128},
}


class PolarNetSeg(nn.Module):
    """PolarNet semantic segmentation (compact): polar BEV projection (theta, r) -> 2D UNet."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        h: int,
        w: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = GridSpec2D(
            x_min=-math.pi,
            x_max=math.pi,
            y_min=0.0,
            y_max=20.0,
            h=int(h),
            w=int(w),
        )
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet = TinyUNet2D(int(width), int(width))
        self.cls = nn.Sequential(
            nn.Linear(int(width), int(width)),
            nn.ReLU(inplace=True),
            nn.Linear(int(width), int(num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x.to(torch.float32))

        theta = torch.atan2(xyz[..., 1], xyz[..., 0])
        r = xyz[..., :2].norm(dim=-1)
        idx = self.grid.quantize(torch.stack([theta, r], dim=-1))
        bev = scatter_mean_2d(idx, p, h=int(self.grid.h), w=int(self.grid.w))
        feat = self.unet(bev)
        gathered = gather_2d(feat, idx)
        return self.cls(gathered)


def build_polarnet_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "polarnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PolarNetSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        h=int(cfg["h"]),
        w=int(cfg["w"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_polarnet_segmenter3d(in_channels=3, num_classes=6, variant="polarnet_tiny")
    x = torch.randn(2, 256, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
