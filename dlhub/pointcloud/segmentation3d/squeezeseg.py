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
    "squeezeseg_tiny": {"width": 48, "h": 32, "w": 96},
    "squeezeseg_small": {"width": 64, "h": 48, "w": 128},
    "squeezeseg_base": {"width": 96, "h": 64, "w": 160},
}


class _SE2D(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        w = int(width)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(w, max(8, w // 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, w // 8), w, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SqueezeSeg(nn.Module):
    """SqueezeSeg semantic segmentation (compact): range-view UNet + squeeze-excitation."""

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
            y_min=-0.5 * math.pi,
            y_max=0.5 * math.pi,
            h=int(h),
            w=int(w),
        )
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet = TinyUNet2D(int(width), int(width))
        self.se = _SE2D(int(width))
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

        az = torch.atan2(xyz[..., 1], xyz[..., 0])
        el = torch.atan2(xyz[..., 2], (xyz[..., :2].norm(dim=-1) + 1e-6))
        idx = self.grid.quantize(torch.stack([az, el], dim=-1))
        rv = scatter_mean_2d(idx, p, h=int(self.grid.h), w=int(self.grid.w))
        feat = self.se(self.unet(rv))
        gathered = gather_2d(feat, idx)
        return self.cls(gathered)


def build_squeezeseg_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "squeezeseg_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return SqueezeSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        h=int(cfg["h"]),
        w=int(cfg["w"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_squeezeseg_segmenter3d(in_channels=3, num_classes=6, variant="squeezeseg_tiny")
    x = torch.randn(2, 256, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
