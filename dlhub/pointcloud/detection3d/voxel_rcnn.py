from __future__ import annotations

import torch
from torch import nn

from ._common import BEVBoxSpec, BEVTwoStageDetector3D, roi_pool_knn, scatter_mean_2d, split_xyz_features, check_points


_VARIANTS: dict[str, dict[str, object]] = {
    "voxel_rcnn_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 48, "roi_k": 8},
    "voxel_rcnn_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 64, "roi_k": 16},
    "voxel_rcnn_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 96, "roi_k": 24},
}


class VoxelRCNN(nn.Module):
    """Voxel R-CNN (toy): two-stage BEV proposals + voxel/point pooled refinement."""

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
        self.det = BEVTwoStageDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            bev=BEVBoxSpec(h=int(bev_h), w=int(bev_w)),
            topk=int(topk),
            roi_k=int(roi_k),
            with_yaw=True,
            dropout=float(dropout),
        )
        # A tiny extra head that mixes stage1 BEV features with ROI pooled point features.
        d = int(width)
        self.mix = nn.Sequential(nn.Linear(d * 2, d), nn.ReLU(inplace=True), nn.Linear(d, d), nn.ReLU(inplace=True))
        self.cls = nn.Linear(d, int(num_classes))
        self.box = nn.Linear(d, 7)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        # Reuse stage1 features explicitly (toy "voxel ROI pooling").
        check_points(points)
        xyz, feats = split_xyz_features(points)
        if feats is None:
            feats = xyz
        x = torch.cat([xyz, feats], dim=-1) if feats is not xyz else xyz
        p = self.det.stage1.point(x)

        out = self.det(points)
        boxes = out["boxes"]
        centers = boxes[..., :3]
        roi = roi_pool_knn(xyz, p, centers, k=self.det.roi_k)
        # A coarse BEV-pooled feature: mean of point features in neighborhood again (proxy).
        bev_like = roi_pool_knn(xyz, p, centers, k=max(4, self.det.roi_k // 2))

        mixed = self.mix(torch.cat([roi, bev_like], dim=-1))
        cls_logits = self.cls(mixed)
        delta = self.box(mixed).tanh()
        boxes2 = boxes + 0.1 * delta
        return {"boxes": boxes2, "cls_logits": cls_logits, "scores": out.get("scores")}


def build_voxel_rcnn_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "voxel_rcnn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return VoxelRCNN(
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
    m = build_voxel_rcnn_detector3d(in_channels=3, num_classes=3, variant="voxel_rcnn_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

