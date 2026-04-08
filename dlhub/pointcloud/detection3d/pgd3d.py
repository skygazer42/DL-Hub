import torch
from torch import nn

from ._common import BEVAnchorFreeDetector3D, BEVBoxSpec

_VARIANTS: dict[str, dict[str, int]] = {
    "pgd3d_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 48},
    "pgd3d_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 64},
    "pgd3d_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 96},
}


class Pgd3dDetector3D(nn.Module):
    """Toy 3D detector for the pgd3d family."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, bev_h: int, bev_w: int, topk: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.det = BEVAnchorFreeDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            bev=BEVBoxSpec(h=int(bev_h), w=int(bev_w)),
            topk=int(topk),
            with_yaw=True,
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.det(points)


def build_pgd3d_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pgd3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return Pgd3dDetector3D(
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
    m = build_pgd3d_detector3d(in_channels=3, num_classes=3, variant="pgd3d_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
