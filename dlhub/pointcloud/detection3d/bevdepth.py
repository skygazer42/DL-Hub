import torch
from torch import nn

from ._common import BEVBoxSpec, BEVTwoStageDetector3D

_VARIANTS: dict[str, dict[str, int]] = {
    "bevdepth_tiny": {"width": 64, "bev_h": 24, "bev_w": 24, "topk": 48, "roi_k": 8},
    "bevdepth_small": {"width": 96, "bev_h": 32, "bev_w": 32, "topk": 64, "roi_k": 16},
    "bevdepth_base": {"width": 128, "bev_h": 40, "bev_w": 40, "topk": 96, "roi_k": 24},
}


class BevdepthDetector3D(nn.Module):
    """Toy two-stage 3D detector for the bevdepth family."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, bev_h: int, bev_w: int, topk: int, roi_k: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.det = BEVTwoStageDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            bev=BEVBoxSpec(h=int(bev_h), w=int(bev_w)),
            topk=int(topk),
            roi_k=int(roi_k),
            with_yaw=True,
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.det(points)


def build_bevdepth_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "bevdepth_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return BevdepthDetector3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        topk=int(cfg["topk"]),
        roi_k=int(cfg["roi_k"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_bevdepth_detector3d(in_channels=3, num_classes=3, variant="bevdepth_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
