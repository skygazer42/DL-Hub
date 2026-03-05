from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ._common import (
    BEVBoxSpec,
    DenseBEVHead,
    PointNetEncoder,
    TinyBEVBackbone,
    check_points,
    decode_bev_boxes,
    scatter_mean_2d,
    split_xyz_features,
    topk_heatmap,
)


_VARIANTS: dict[str, dict[str, object]] = {
    "voxelnext_tiny": {"width": 64, "bev_h": 32, "bev_w": 32, "topk": 64},
    "voxelnext_small": {"width": 96, "bev_h": 40, "bev_w": 40, "topk": 96},
    "voxelnext_base": {"width": 128, "bev_h": 48, "bev_w": 48, "topk": 128},
}


class _MSBackbone(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        w = int(width)
        self.stem = TinyBEVBackbone(int(in_channels), width=w)
        self.down = nn.Conv2d(w, w, 3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(w, w, 4, stride=2, padding=1)
        self.fuse = nn.Conv2d(w * 2, w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        lo = F.relu(self.down(x), inplace=True)
        hi = F.relu(self.up(lo), inplace=True)
        if hi.shape[-2:] != x.shape[-2:]:
            hi = F.interpolate(hi, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return F.relu(self.fuse(torch.cat([x, hi], dim=1)), inplace=True)


class VoxelNeXt(nn.Module):
    """VoxelNeXt (toy): multi-scale BEV backbone + dense head."""

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

        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.backbone = _MSBackbone(int(width), int(width))
        self.head = DenseBEVHead(int(width), int(num_classes), with_yaw=True)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)

        idx = self.bev.quantize_xy(xyz[..., :2])
        bev = scatter_mean_2d(idx, p, h=int(self.bev.h), w=int(self.bev.w))
        feat = self.backbone(bev)
        dense = self.head(feat)

        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.bev, with_yaw=True)
        cls_logits = torch.zeros(points.shape[0], self.topk, self.num_classes, device=points.device, dtype=points.dtype)
        cls_logits.scatter_(-1, cls.unsqueeze(-1), scores.unsqueeze(-1))
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


def build_voxelnext_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "voxelnext_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return VoxelNeXt(
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
    m = build_voxelnext_detector3d(in_channels=3, num_classes=3, variant="voxelnext_tiny")
    x = torch.randn(2, 512, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

