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
    "second_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 64},
    "second_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 96},
    "second_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 128},
}


class SECOND(nn.Module):
    """SECOND (toy): sparse-ish BEV conv backbone + dense head.

    We keep it CPU-friendly by using a small BEV grid and regular Conv2d.
    """

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

        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.backbone = nn.Sequential(
            TinyBEVBackbone(int(width), width=int(width)),
            nn.Conv2d(int(width), int(width), 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = DenseBEVHead(int(width), int(num_classes), with_yaw=True)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        if feats is None:
            feats = xyz
        x = torch.cat([xyz, feats], dim=-1) if feats is not xyz else xyz

        p = self.point(x)
        idx = self.bev.quantize_xy(xyz[..., :2])
        bev = scatter_mean_2d(idx, p, h=int(self.bev.h), w=int(self.bev.w))

        feat = self.backbone(bev)
        dense = self.head(feat)

        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.bev, with_yaw=True)

        # Map `scores` into per-class logits for a stable shape.
        cls_logits = torch.full(
            (points.shape[0], self.topk, dense["heatmap"].shape[1]),
            -10.0,
            device=points.device,
            dtype=points.dtype,
        )
        cls_logits.scatter_(
            -1, cls.unsqueeze(-1), torch.logit(scores.clamp(1e-4, 1 - 1e-4)).unsqueeze(-1)
        )
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


def build_second_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "second_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return SECOND(
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
    m = build_second_detector3d(in_channels=3, num_classes=3, variant="second_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
