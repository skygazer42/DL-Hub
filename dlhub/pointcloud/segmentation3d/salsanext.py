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
    "salsanext_tiny": {"width": 48, "h": 32, "w": 96},
    "salsanext_small": {"width": 64, "h": 48, "w": 128},
    "salsanext_base": {"width": 96, "h": 64, "w": 160},
}


class _Res2D(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        w = int(width)
        self.net = nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv2d(w, w, 3, padding=1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SalsaNextSeg(nn.Module):
    """SalsaNext semantic segmentation (toy): range-view UNet + residual refinement."""

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
        self.refine = _Res2D(int(width), float(dropout))
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
        feat = self.refine(self.unet(rv))
        gathered = gather_2d(feat, idx)
        return self.cls(gathered)


def build_salsanext_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "salsanext_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return SalsaNextSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        h=int(cfg["h"]),
        w=int(cfg["w"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_salsanext_segmenter3d(in_channels=3, num_classes=6, variant="salsanext_tiny")
    x = torch.randn(2, 256, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
