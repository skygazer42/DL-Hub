
import torch
from torch import nn

from ._common import (
    BEVBoxSpec,
    DenseBEVHead,
    PointNetEncoder,
    TinyTransformerEncoder,
    check_points,
    decode_bev_boxes,
    scatter_mean_2d,
    split_xyz_features,
    topk_heatmap,
)


_VARIANTS: dict[str, dict[str, object]] = {
    "votr_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 64, "layers": 1},
    "votr_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 96, "layers": 2},
    "votr_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 128, "layers": 3},
}


class VoTR(nn.Module):
    """VoTR (toy): voxel/BEV tokens processed by a transformer encoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev_h: int,
        bev_w: int,
        topk: int,
        layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bev = BEVBoxSpec(h=int(bev_h), w=int(bev_w))
        self.topk = int(topk)

        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.token_proj = nn.Linear(int(width), int(width))
        self.enc = TinyTransformerEncoder(int(width), nhead=4, num_layers=int(layers), dropout=float(dropout))
        self.head = DenseBEVHead(int(width), int(num_classes), with_yaw=True)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)  # (B,N,D)

        idx = self.bev.quantize_xy(xyz[..., :2])
        bev = scatter_mean_2d(idx, p, h=int(self.bev.h), w=int(self.bev.w))  # (B,D,H,W)

        b, d, h, w = bev.shape
        tokens = bev.permute(0, 2, 3, 1).reshape(b, h * w, d)
        tokens = self.token_proj(tokens)
        tokens = self.enc(tokens)
        feat = tokens.reshape(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        dense = self.head(feat)
        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.bev, with_yaw=True)

        cls_logits = torch.zeros(points.shape[0], self.topk, dense["heatmap"].shape[1], device=points.device, dtype=points.dtype)
        cls_logits.scatter_(-1, cls.unsqueeze(-1), scores.unsqueeze(-1))
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


def build_votr_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "votr_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return VoTR(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        topk=int(cfg["topk"]),
        layers=int(cfg["layers"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_votr_detector3d(in_channels=3, num_classes=3, variant="votr_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

