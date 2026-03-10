import torch
from torch import nn

from ._common import BEVAnchorFreeDetector3D, BEVBoxSpec

_VARIANTS: dict[str, dict[str, object]] = {
    "pixor_tiny": {"width": 48, "bev_h": 48, "bev_w": 48, "topk": 64},
    "pixor_small": {"width": 64, "bev_h": 64, "bev_w": 64, "topk": 96},
    "pixor_base": {"width": 96, "bev_h": 80, "bev_w": 80, "topk": 128},
}


class PIXOR(nn.Module):
    """PIXOR (toy): BEV-only detector with tight 2D boxes + yaw."""

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
        # PIXOR is BEV-focused; we keep a larger grid by default.
        self.det = BEVAnchorFreeDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            bev=BEVBoxSpec(
                h=int(bev_h), w=int(bev_w), x_min=-20.0, x_max=20.0, y_min=-20.0, y_max=20.0
            ),
            topk=int(topk),
            with_yaw=True,
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.det(points)


def build_pixor_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pixor_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PIXOR(
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
    m = build_pixor_detector3d(in_channels=3, num_classes=3, variant="pixor_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
