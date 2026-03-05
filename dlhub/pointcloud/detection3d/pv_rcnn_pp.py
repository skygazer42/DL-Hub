from __future__ import annotations

import torch
from torch import nn

import math

from ._common import BEVBoxSpec, BEVTwoStageDetector3D, check_points, mlp, roi_pool_knn, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "pv_rcnn_pp_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 48, "roi_k": 8, "refine_depth": 1},
    "pv_rcnn_pp_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 64, "roi_k": 16, "refine_depth": 2},
    "pv_rcnn_pp_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 96, "roi_k": 24, "refine_depth": 3},
}


class PVRCNNPlusPlus(nn.Module):
    """PV-RCNN++ (toy): stronger ROI refinement (stacked MLP)."""

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
        refine_depth: int,
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

        d = int(width)
        blocks: list[nn.Module] = [mlp(d, [d, d], d, dropout=float(dropout)) for _ in range(int(refine_depth))]
        self.extra_refine = nn.Sequential(*blocks) if blocks else nn.Identity()

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        # Re-run the two-stage refinement with an extra MLP stack (toy "++").
        check_points(points)
        xyz, feats = split_xyz_features(points)
        if feats is None:
            feats = xyz
        x = torch.cat([xyz, feats], dim=-1) if feats is not xyz else xyz
        p = self.det.stage1.point(x)

        out1 = self.det.stage1(points)
        boxes = out1["boxes"]
        pooled = roi_pool_knn(xyz, p, boxes[..., :3], k=self.det.roi_k)
        r = self.det.refine(pooled)
        r = self.extra_refine(r)

        cls_logits = self.det.cls(r)
        raw = self.det.box(r)

        delta_xyz = raw[..., :3].tanh()
        delta_dims = raw[..., 3:6].tanh()
        new_xyz = boxes[..., :3] + delta_xyz
        new_dims = (boxes[..., 3:6] * (1.0 + 0.1 * delta_dims)).clamp_min(0.05)
        yaw_base = boxes[..., 6:7] if boxes.shape[-1] == 7 else torch.zeros_like(raw[..., 6:7])
        new_yaw = (yaw_base + raw[..., 6:7].tanh() * 0.1).clamp(-math.pi, math.pi)
        boxes2 = torch.cat([new_xyz, new_dims, new_yaw], dim=-1)

        return {"boxes": boxes2, "cls_logits": cls_logits, "scores": out1.get("scores")}


def build_pv_rcnn_pp_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pv_rcnn_pp_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PVRCNNPlusPlus(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        topk=int(cfg["topk"]),
        roi_k=int(cfg["roi_k"]),
        refine_depth=int(cfg["refine_depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_pv_rcnn_pp_detector3d(in_channels=3, num_classes=3, variant="pv_rcnn_pp_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
