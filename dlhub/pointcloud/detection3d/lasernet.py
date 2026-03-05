from __future__ import annotations

import math

import torch
from torch import nn

from ._common import DenseBEVHead, PointNetEncoder, TinyBEVBackbone, check_points, decode_bev_boxes, scatter_mean_2d, split_xyz_features, topk_heatmap, BEVBoxSpec


_VARIANTS: dict[str, dict[str, object]] = {
    "lasernet_tiny": {"width": 48, "h": 32, "w": 64, "topk": 48},
    "lasernet_small": {"width": 64, "h": 48, "w": 96, "topk": 64},
    "lasernet_base": {"width": 96, "h": 64, "w": 128, "topk": 96},
}


class LaserNet(nn.Module):
    """LaserNet (toy): range-view style projection + 2D conv head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        h: int,
        w: int,
        topk: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rv = BEVBoxSpec(
            h=int(h),
            w=int(w),
            x_min=-math.pi,
            x_max=math.pi,
            y_min=-0.4 * math.pi,
            y_max=0.4 * math.pi,
        )
        self.topk = int(topk)
        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.backbone = TinyBEVBackbone(int(width), width=int(width))
        self.head = DenseBEVHead(int(width), int(num_classes), with_yaw=True)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)

        # Project to range view by angles (azimuth/elevation).
        az = torch.atan2(xyz[..., 1], xyz[..., 0])
        el = torch.atan2(xyz[..., 2], (xyz[..., :2].norm(dim=-1) + 1e-6))
        idx = self.rv.quantize_xy(torch.stack([az, el], dim=-1))
        rv = scatter_mean_2d(idx, p, h=int(self.rv.h), w=int(self.rv.w))

        feat = self.backbone(rv)
        dense = self.head(feat)
        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.rv, with_yaw=True)

        cls_logits = torch.zeros(points.shape[0], self.topk, dense["heatmap"].shape[1], device=points.device, dtype=points.dtype)
        cls_logits.scatter_(-1, cls.unsqueeze(-1), scores.unsqueeze(-1))
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


def build_lasernet_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "lasernet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return LaserNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        h=int(cfg["h"]),
        w=int(cfg["w"]),
        topk=int(cfg["topk"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_lasernet_detector3d(in_channels=3, num_classes=3, variant="lasernet_tiny")
    x = torch.randn(2, 512, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

