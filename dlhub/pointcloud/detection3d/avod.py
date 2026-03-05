from __future__ import annotations

import torch
from torch import nn

from ._common import (
    BEVBoxSpec,
    BEVTwoStageDetector3D,
    PointNetEncoder,
    TinyBEVBackbone,
    check_points,
    roi_pool_knn,
    scatter_mean_2d,
    split_xyz_features,
    topk_heatmap,
    decode_bev_boxes,
    DenseBEVHead,
)


_VARIANTS: dict[str, dict[str, object]] = {
    "avod_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 48, "roi_k": 8},
    "avod_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 64, "roi_k": 16},
    "avod_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 96, "roi_k": 24},
}


class AVOD(nn.Module):
    """AVOD (toy): fuses a coarse+fine BEV backbone before ROI refinement."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev_h: int,
        bev_w: int,
        topk: int,
        roi_k: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bev = BEVBoxSpec(h=int(bev_h), w=int(bev_w))
        self.topk = int(topk)
        self.roi_k = int(roi_k)
        self.num_classes = int(num_classes)

        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.backbone_fine = TinyBEVBackbone(int(width), width=int(width))
        self.backbone_coarse = nn.Sequential(
            nn.Conv2d(int(width), int(width), 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(width), int(width), 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(int(width) * 2, int(width), 1)
        self.head = DenseBEVHead(int(width), int(num_classes), with_yaw=True)

        d = int(width)
        self.roi = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.cls = nn.Linear(d, int(num_classes))
        self.box = nn.Linear(d, 7)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)
        idx = self.bev.quantize_xy(xyz[..., :2])
        bev = scatter_mean_2d(idx, p, h=int(self.bev.h), w=int(self.bev.w))

        fine = self.backbone_fine(bev)
        coarse = self.backbone_coarse(bev)
        feat = self.fuse(torch.cat([fine, coarse], dim=1))
        dense = self.head(feat)

        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.bev, with_yaw=True)

        pooled = roi_pool_knn(xyz, p, boxes[..., :3], k=self.roi_k)
        r = self.roi(pooled)
        cls_logits = self.cls(r)
        boxes = boxes + 0.1 * self.box(r).tanh()
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


def build_avod_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "avod_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return AVOD(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        topk=int(cfg["topk"]),
        roi_k=int(cfg["roi_k"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_avod_detector3d(in_channels=3, num_classes=3, variant="avod_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

