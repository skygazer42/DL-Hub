from __future__ import annotations

import torch
from torch import nn

from ._common import PointQueryDetector3D, check_points


_VARIANTS: dict[str, dict[str, object]] = {
    "imvotenet_tiny": {"d_model": 64, "queries": 32},
    "imvotenet_small": {"d_model": 96, "queries": 48},
    "imvotenet_base": {"d_model": 128, "queries": 64},
}


class ImVoteNet(nn.Module):
    """ImVoteNet (toy): query-based detector (image fusion omitted in toy interface)."""

    def __init__(self, *, in_channels: int, num_classes: int, d_model: int, num_queries: int, dropout: float = 0.0) -> None:
        super().__init__()
        # In real ImVoteNet, image cues influence voting. Here we keep a DETR-like head.
        self.det = PointQueryDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            d_model=int(d_model),
            num_queries=int(num_queries),
            use_transformer=True,
            dropout=float(dropout),
            with_yaw=True,
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        return self.det(points)


def build_imvotenet_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "imvotenet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return ImVoteNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_imvotenet_detector3d(in_channels=3, num_classes=5, variant="imvotenet_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

