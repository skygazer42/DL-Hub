from __future__ import annotations

import torch
from torch import nn

from ._common import GridSpec2D, PointMLP, TinyUNet2D, check_points, gather_2d, scatter_mean_2d, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "simpleview_tiny": {"width": 48, "hw": 24},
    "simpleview_small": {"width": 64, "hw": 32},
    "simpleview_base": {"width": 96, "hw": 40},
}


class SimpleViewSeg(nn.Module):
    """SimpleView semantic segmentation (toy): fuse two orthographic projections (XY + XZ)."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, hw: int, dropout: float = 0.0) -> None:
        super().__init__()
        h = int(hw)
        w = int(hw)
        self.xy = GridSpec2D(h=h, w=w)
        self.xz = GridSpec2D(h=h, w=w, y_min=-2.0, y_max=2.0)
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet_xy = TinyUNet2D(int(width), int(width))
        self.unet_xz = TinyUNet2D(int(width), int(width))
        self.fuse = nn.Sequential(nn.Linear(int(width) * 2, int(width)), nn.ReLU(inplace=True))
        self.cls = nn.Sequential(nn.Linear(int(width), int(width)), nn.ReLU(inplace=True), nn.Linear(int(width), int(num_classes)))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x.to(torch.float32))

        idx_xy = self.xy.quantize(xyz[..., :2])
        bev_xy = scatter_mean_2d(idx_xy, p, h=int(self.xy.h), w=int(self.xy.w))
        f_xy = self.unet_xy(bev_xy)
        g_xy = gather_2d(f_xy, idx_xy)

        idx_xz = self.xz.quantize(torch.stack([xyz[..., 0], xyz[..., 2]], dim=-1))
        bev_xz = scatter_mean_2d(idx_xz, p, h=int(self.xz.h), w=int(self.xz.w))
        f_xz = self.unet_xz(bev_xz)
        g_xz = gather_2d(f_xz, idx_xz)

        y = self.fuse(torch.cat([g_xy, g_xz], dim=-1))
        return self.cls(y)


def build_simpleview_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "simpleview_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return SimpleViewSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        hw=int(cfg["hw"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_simpleview_segmenter3d(in_channels=3, num_classes=6, variant="simpleview_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

