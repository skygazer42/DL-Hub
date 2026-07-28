import math

import torch
from torch import nn

from ._common import QueryMaskHead, TransformerPointEncoder, l2_normalize

_VARIANTS: dict[str, dict[str, object]] = {
    "pointtrackmask3d_tiny": {"d_model": 64, "depth": 2, "queries": 16},
    "pointtrackmask3d_small": {"d_model": 96, "depth": 3, "queries": 24},
    "pointtrackmask3d_base": {"d_model": 128, "depth": 4, "queries": 32},
}


class PointtrackMask3D(nn.Module):
    """PointtrackMask3D (compact): initial queries, then refine by pooling point features with masks."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        d_model: int,
        depth: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(d_model)
        self.enc = TransformerPointEncoder(
            int(in_channels), d, depth=int(depth), dropout=float(dropout)
        )
        self.head1 = QueryMaskHead(
            d, int(num_classes), num_queries=int(num_queries), dropout=float(dropout)
        )
        self.cls2 = nn.Linear(d, int(num_classes))
        self.q2 = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)  # (B,N,D)
        out1 = self.head1(xyz, feat)
        m1 = out1["mask_logits"].sigmoid()  # (B,K,N)
        w = m1 / m1.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        inst_feat = torch.einsum("bkn,bnd->bkd", w, feat)  # (B,K,D)

        inst_feat2 = inst_feat + 0.1 * self.q2(inst_feat).tanh()
        d = feat.shape[-1]
        mask_logits = torch.einsum(
            "bkd,bnd->bkn", l2_normalize(inst_feat2), l2_normalize(feat)
        ) * math.sqrt(d)
        cls_logits = self.cls2(inst_feat2)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits}


def build_pointtrackmask3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointtrackmask3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return PointtrackMask3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        depth=int(cfg["depth"]),
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_pointtrackmask3d_instance_segmenter3d(
        in_channels=3, num_classes=6, variant="pointtrackmask3d_tiny"
    )
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
