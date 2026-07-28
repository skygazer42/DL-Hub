import math

import torch
from torch import nn

from ._common import (
    GridSpec2D,
    PointMLP,
    TinyTransformerEncoder,
    check_points,
    gather_2d,
    scatter_mean_2d,
    split_xyz_features,
)

_VARIANTS: dict[str, dict[str, object]] = {
    "sphereformer_tiny": {"width": 64, "depth": 2, "h": 32, "w": 64},
    "sphereformer_small": {"width": 96, "depth": 3, "h": 48, "w": 96},
    "sphereformer_base": {"width": 128, "depth": 4, "h": 64, "w": 128},
}


class SphereFormerSeg(nn.Module):
    """SphereFormer semantic segmentation (compact spherical projection encoder)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
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
        self.enc = TinyTransformerEncoder(
            int(width), nhead=4, num_layers=int(depth), dropout=float(dropout)
        )
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
        phi = torch.atan2(xyz[..., 2], xyz.norm(dim=-1) + 1e-6)
        idx = self.grid.quantize(torch.stack([theta, phi], dim=-1))
        sph = scatter_mean_2d(idx, p, h=int(self.grid.h), w=int(self.grid.w))
        b, c, h, w = sph.shape
        tok = sph.permute(0, 2, 3, 1).reshape(b, h * w, c)
        tok = self.enc(tok)
        feat = tok.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        gathered = gather_2d(feat, idx)
        return self.cls(gathered)


def build_sphereformer_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "sphereformer_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    return SphereFormerSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(int(cfg["width"]) * float(width_mult)),
        depth=int(cfg["depth"]),
        h=int(cfg["h"]),
        w=int(cfg["w"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_sphereformer_segmenter3d(
        in_channels=3, num_classes=6, variant="sphereformer_tiny"
    )
    x = torch.randn(2, 256, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
