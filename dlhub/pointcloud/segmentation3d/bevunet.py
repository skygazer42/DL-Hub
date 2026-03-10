import torch
from torch import nn

from ._common import GridSpec2D, Projection2DSegBase

_VARIANTS: dict[str, dict[str, object]] = {
    "bevunet_tiny": {"width": 48, "bev_h": 24, "bev_w": 24},
    "bevunet_small": {"width": 64, "bev_h": 32, "bev_w": 32},
    "bevunet_base": {"width": 96, "bev_h": 40, "bev_w": 40},
}


class BEVUNetSeg(nn.Module):
    """BEV UNet semantic segmentation (toy): XY BEV projection + 2D UNet."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev_h: int,
        bev_w: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = Projection2DSegBase(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            grid=GridSpec2D(h=int(bev_h), w=int(bev_w)),
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def build_bevunet_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "bevunet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return BEVUNetSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_bevunet_segmenter3d(in_channels=3, num_classes=6, variant="bevunet_tiny")
    x = torch.randn(2, 256, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
