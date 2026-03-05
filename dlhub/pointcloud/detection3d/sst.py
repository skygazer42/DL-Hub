from __future__ import annotations

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
    "sst_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 64, "layers": 1},
    "sst_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 96, "layers": 2},
    "sst_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 128, "layers": 3},
}


class SST(nn.Module):
    """SST (toy): sparse-ish spatial transformer on BEV tokens.

    We approximate sparsity by selecting top occupied cells (based on BEV magnitude),
    running transformer on them, then writing back.
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
        layers: int,
        select_k: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bev = BEVBoxSpec(h=int(bev_h), w=int(bev_w))
        self.topk = int(topk)
        self.select_k = int(select_k)

        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.enc = TinyTransformerEncoder(int(width), nhead=4, num_layers=int(layers), dropout=float(dropout))
        self.head = DenseBEVHead(int(width), int(num_classes), with_yaw=True)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)

        idx = self.bev.quantize_xy(xyz[..., :2])
        bev = scatter_mean_2d(idx, p, h=int(self.bev.h), w=int(self.bev.w))  # (B,D,H,W)

        b, d, h, w = bev.shape
        mag = bev.abs().mean(dim=1)  # (B,H,W)
        flat_mag = mag.view(b, h * w)
        k = min(self.select_k, h * w)
        sel = flat_mag.topk(k, dim=-1).indices  # (B,k)

        tokens = bev.permute(0, 2, 3, 1).reshape(b, h * w, d)
        sel_tokens = tokens.gather(1, sel.unsqueeze(-1).expand(b, k, d))
        sel_tokens = self.enc(sel_tokens)

        tokens2 = tokens.clone()
        tokens2.scatter_(1, sel.unsqueeze(-1).expand(b, k, d), sel_tokens)
        feat = tokens2.reshape(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        dense = self.head(feat)
        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.bev, with_yaw=True)

        cls_logits = torch.zeros(points.shape[0], self.topk, dense["heatmap"].shape[1], device=points.device, dtype=points.dtype)
        cls_logits.scatter_(-1, cls.unsqueeze(-1), scores.unsqueeze(-1))
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


def build_sst_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "sst_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    select_k = max(64, int(int(cfg["bev_h"]) * int(cfg["bev_w"]) * 0.25))
    return SST(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        topk=int(cfg["topk"]),
        layers=int(cfg["layers"]),
        select_k=int(select_k),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_sst_detector3d(in_channels=3, num_classes=3, variant="sst_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

