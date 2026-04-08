import math

import torch
from torch import nn

from ._common import QueryMaskHead, TransformerPointEncoder, l2_normalize

_VARIANTS: dict[str, dict[str, object]] = {
    "pointmamba_inst_tiny": {"d_model": 64, "depth": 2, "queries": 16},
    "pointmamba_inst_small": {"d_model": 96, "depth": 3, "queries": 24},
    "pointmamba_inst_base": {"d_model": 128, "depth": 4, "queries": 32},
}


class PointMambaInst(nn.Module):
    """PointMamba-Inst (toy): query refinement over point tokens for instance masks."""

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
        xyz, feat = self.enc(points)
        out1 = self.head1(xyz, feat)
        mask_prob = out1["mask_logits"].sigmoid()
        weights = mask_prob / mask_prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        inst_feat = torch.einsum("bkn,bnd->bkd", weights, feat)
        refined = inst_feat + 0.1 * self.q2(inst_feat).tanh()
        scale = math.sqrt(feat.shape[-1])
        mask_logits = torch.einsum(
            "bkd,bnd->bkn", l2_normalize(refined), l2_normalize(feat)
        ) * scale
        cls_logits = self.cls2(refined)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits}


def build_pointmamba_inst_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointmamba_inst_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return PointMambaInst(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        depth=int(cfg["depth"]),
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_pointmamba_inst_instance_segmenter3d(
        in_channels=3,
        num_classes=6,
        variant="pointmamba_inst_tiny",
    )
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
