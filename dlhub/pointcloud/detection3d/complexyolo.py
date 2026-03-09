
import math

import torch
from torch import nn
from torch.nn import functional as F

from ._common import (
    BEVBoxSpec,
    PointNetEncoder,
    TinyBEVBackbone,
    check_points,
    decode_bev_boxes,
    scatter_mean_2d,
    split_xyz_features,
    topk_heatmap,
)


_VARIANTS: dict[str, dict[str, object]] = {
    "complexyolo_tiny": {"width": 64, "bev_h": 32, "bev_w": 32, "topk": 64},
    "complexyolo_small": {"width": 96, "bev_h": 40, "bev_w": 40, "topk": 96},
    "complexyolo_base": {"width": 128, "bev_h": 48, "bev_w": 48, "topk": 128},
}


class _ComplexYawHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.hm = nn.Conv2d(int(in_channels), int(num_classes), 1)
        # x,y,z, dx,dy,dz, sin(yaw), cos(yaw)
        self.box = nn.Conv2d(int(in_channels), 8, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"heatmap": self.hm(x), "box_params": self.box(x)}


class ComplexYOLO(nn.Module):
    """Complex-YOLO (toy): yaw via complex angle (sin/cos) on BEV grid."""

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
        self.backbone = TinyBEVBackbone(int(width), width=int(width))
        self.head = _ComplexYawHead(int(width), int(num_classes))

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
        params = dense["box_params"]  # (B,8,H,W)
        gathered = params[:, :6]  # xyz+dims
        boxes6 = decode_bev_boxes(gathered, iy, ix, self.bev, with_yaw=False)
        sc = torch.stack(
            [
                dense["box_params"][:, 6:7],
                dense["box_params"][:, 7:8],
            ],
            dim=1,
        )  # (B,2,1,H,W)
        sc = sc.squeeze(2)  # (B,2,H,W)
        sc_g = sc.permute(0, 2, 3, 1).reshape(sc.shape[0], -1, 2)
        # Gather sin/cos using same flat indices.
        flat = (iy * int(self.bev.w) + ix).to(torch.long)  # (B,K)
        sincos = sc_g.gather(1, flat.unsqueeze(-1).expand(-1, -1, 2))
        yaw = torch.atan2(sincos[..., 0:1], sincos[..., 1:2]).clamp(-math.pi, math.pi)
        boxes = torch.cat([boxes6, yaw], dim=-1)

        cls_logits = torch.zeros(points.shape[0], self.topk, self.num_classes, device=points.device, dtype=points.dtype)
        cls_logits.scatter_(-1, cls.unsqueeze(-1), scores.unsqueeze(-1))
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


def build_complexyolo_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "complexyolo_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return ComplexYOLO(
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
    m = build_complexyolo_detector3d(in_channels=3, num_classes=3, variant="complexyolo_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

