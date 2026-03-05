from __future__ import annotations

import torch
from torch import nn

from ._common import BEVBoxSpec, DenseBEVHead, PointNetEncoder, TinyBEVBackbone, check_points, decode_bev_boxes, scatter_mean_2d, split_xyz_features, topk_heatmap


_VARIANTS: dict[str, dict[str, object]] = {
    "bevfusion_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 48},
    "bevfusion_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 64},
    "bevfusion_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 96},
}


class BEVFusion(nn.Module):
    """BEVFusion (toy): fuse two BEV feature maps then detect."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev_h: int,
        bev_w: int,
        topk: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bev = BEVBoxSpec(h=int(bev_h), w=int(bev_w))
        self.topk = int(topk)
        self.num_classes = int(num_classes)

        self.point_a = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.point_b = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.backbone_a = TinyBEVBackbone(int(width), width=int(width))
        self.backbone_b = TinyBEVBackbone(int(width), width=int(width))
        self.fuse = nn.Conv2d(int(width) * 2, int(width), 1)
        self.head = DenseBEVHead(int(width), int(num_classes), with_yaw=True)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)

        pa = self.point_a(x)
        pb = self.point_b(x)
        idx = self.bev.quantize_xy(xyz[..., :2])
        bev_a = scatter_mean_2d(idx, pa, h=int(self.bev.h), w=int(self.bev.w))
        bev_b = scatter_mean_2d(idx, pb, h=int(self.bev.h), w=int(self.bev.w))

        fa = self.backbone_a(bev_a)
        fb = self.backbone_b(bev_b)
        feat = self.fuse(torch.cat([fa, fb], dim=1))
        dense = self.head(feat)

        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.bev, with_yaw=True)
        cls_logits = torch.zeros(points.shape[0], self.topk, self.num_classes, device=points.device, dtype=points.dtype)
        cls_logits.scatter_(-1, cls.unsqueeze(-1), scores.unsqueeze(-1))
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


def build_bevfusion_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "bevfusion_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return BEVFusion(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        topk=int(cfg["topk"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_bevfusion_detector3d(in_channels=3, num_classes=3, variant="bevfusion_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

