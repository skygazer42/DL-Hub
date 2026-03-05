from __future__ import annotations

import torch
from torch import nn

from ._common import CenterProposalHead, EdgeConvEncoder


_VARIANTS: dict[str, dict[str, object]] = {
    "hais_tiny": {"width": 64, "depth": 2, "k": 8, "instances": 16},
    "hais_small": {"width": 96, "depth": 3, "k": 16, "instances": 24},
    "hais_base": {"width": 128, "depth": 4, "k": 24, "instances": 32},
}


class HAIS(nn.Module):
    """HAIS (toy): hierarchical gating applied to center-proposal instance masks."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        k: int,
        num_instances: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = EdgeConvEncoder(int(in_channels), int(width), depth=int(depth), k=int(k), dropout=float(dropout))
        self.head = CenterProposalHead(int(width), int(num_classes), num_instances=int(num_instances), dropout=float(dropout))
        self.gate = nn.Sequential(nn.Linear(3, int(width) // 2), nn.ReLU(inplace=True), nn.Linear(int(width) // 2, 1), nn.Sigmoid())

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        out = self.head(xyz, feat)
        centers = out.get("centers")
        if isinstance(centers, torch.Tensor):
            g = self.gate(centers.to(out["mask_logits"].dtype))
            out["mask_logits"] = out["mask_logits"] * (0.5 + 0.5 * g)
        return {"mask_logits": out["mask_logits"], "cls_logits": out["cls_logits"]}


def build_hais_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "hais_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return HAIS(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        k=int(cfg["k"]),
        num_instances=int(cfg["instances"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_hais_instance_segmenter3d(in_channels=3, num_classes=6, variant="hais_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

