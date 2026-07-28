import torch
from torch import nn

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
    "mv3d_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 48},
    "mv3d_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 64},
    "mv3d_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 96},
}


class MV3D(nn.Module):
    """MV3D (compact): fuse BEV (x,y) and front-view (x,z) feature maps."""

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
        # Front-view uses x (w) and z (h)
        self.fv = BEVBoxSpec(
            h=int(bev_h), w=int(bev_w), x_min=-10.0, x_max=10.0, y_min=-3.0, y_max=3.0
        )
        self.topk = int(topk)

        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.bev_backbone = TinyBEVBackbone(int(width), width=int(width))
        self.fv_backbone = TinyBEVBackbone(int(width), width=int(width))
        self.fuse = nn.Conv2d(int(width) * 2, int(width), 1)
        self.head = DenseBEVHead(int(width), int(num_classes), with_yaw=True)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)

        idx_bev = self.bev.quantize_xy(xyz[..., :2])
        bev = scatter_mean_2d(idx_bev, p, h=int(self.bev.h), w=int(self.bev.w))
        bev_f = self.bev_backbone(bev)

        idx_fv = self.fv.quantize_xy(torch.stack([xyz[..., 0], xyz[..., 2]], dim=-1))
        fv = scatter_mean_2d(idx_fv, p, h=int(self.fv.h), w=int(self.fv.w))
        fv_f = self.fv_backbone(fv)

        feat = self.fuse(torch.cat([bev_f, fv_f], dim=1))
        dense = self.head(feat)

        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.bev, with_yaw=True)
        cls_logits = torch.zeros(
            points.shape[0],
            self.topk,
            dense["heatmap"].shape[1],
            device=points.device,
            dtype=points.dtype,
        )
        cls_logits.scatter_(-1, cls.unsqueeze(-1), scores.unsqueeze(-1))
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


def build_mv3d_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "mv3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return MV3D(
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
    m = build_mv3d_detector3d(in_channels=3, num_classes=3, variant="mv3d_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
