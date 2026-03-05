from __future__ import annotations

import torch
from torch import nn

from ._common import BEVBoxSpec, DenseBEVHead, PointNetEncoder, PointQueryDetector3D, TinyBEVBackbone, TinyTransformerEncoder, check_points, decode_bev_boxes, scatter_mean_2d, split_xyz_features, topk_heatmap


_VARIANTS: dict[str, dict[str, object]] = {
    "transfusion_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 48, "queries": 32},
    "transfusion_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 64, "queries": 48},
    "transfusion_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 96, "queries": 64},
}


class TransFusion(nn.Module):
    """TransFusion (toy): fuse a query head with dense BEV candidates."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev_h: int,
        bev_w: int,
        topk: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bev = BEVBoxSpec(h=int(bev_h), w=int(bev_w))
        self.topk = int(topk)

        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.backbone = TinyBEVBackbone(int(width), width=int(width))
        self.dense_head = DenseBEVHead(int(width), int(num_classes), with_yaw=True)

        self.query_det = PointQueryDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            d_model=int(width),
            num_queries=int(num_queries),
            use_transformer=True,
            dropout=float(dropout),
            with_yaw=True,
        )

        self.fuse = nn.Sequential(nn.Linear(7 + int(num_classes), 32), nn.ReLU(inplace=True), nn.Linear(32, 7 + int(num_classes)))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)

        idx = self.bev.quantize_xy(xyz[..., :2])
        bev = scatter_mean_2d(idx, p, h=int(self.bev.h), w=int(self.bev.w))
        feat = self.backbone(bev)
        dense = self.dense_head(feat)
        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        dense_boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.bev, with_yaw=True)
        dense_logits = torch.zeros(points.shape[0], self.topk, dense["heatmap"].shape[1], device=points.device, dtype=points.dtype)
        dense_logits.scatter_(-1, cls.unsqueeze(-1), scores.unsqueeze(-1))

        q = self.query_det(points)

        # Fuse by concatenating query predictions with the best dense candidate (toy).
        fused_boxes = q["boxes"].clone()
        fused_logits = q["cls_logits"].clone()
        best_idx = dense_logits.max(dim=-1).values.argmax(dim=1)  # (B,)
        b = points.shape[0]
        dense_best_boxes = dense_boxes[torch.arange(b, device=points.device), best_idx]
        dense_best_logits = dense_logits[torch.arange(b, device=points.device), best_idx]

        cat = torch.cat([dense_best_boxes, dense_best_logits], dim=-1)  # (B,7+C)
        delta = self.fuse(cat).tanh()
        fused_boxes[:, :1] = fused_boxes[:, :1] + 0.05 * delta[:, :7].unsqueeze(1)
        fused_logits = fused_logits + 0.05 * delta[:, 7:].unsqueeze(1)

        return {"boxes": fused_boxes, "cls_logits": fused_logits}


def build_transfusion_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "transfusion_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return TransFusion(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        topk=int(cfg["topk"]),
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_transfusion_detector3d(in_channels=3, num_classes=3, variant="transfusion_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

