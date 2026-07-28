import torch
from torch import nn

from ._common import EdgeConvSegBase

_VARIANTS: dict[str, dict[str, object]] = {
    "dgcnn_tiny": {"width": 48, "k": 8, "depth": 2},
    "dgcnn_small": {"width": 64, "k": 16, "depth": 3},
    "dgcnn_base": {"width": 96, "k": 24, "depth": 4},
}


class DGCNNSeg(nn.Module):
    """DGCNN semantic segmentation (compact): EdgeConv stack -> per-point logits."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        k: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = EdgeConvSegBase(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            k=int(k),
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def build_dgcnn_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "dgcnn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return DGCNNSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        k=int(cfg["k"]),
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_dgcnn_segmenter3d(in_channels=3, num_classes=6, variant="dgcnn_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    (y.mean()).backward()
    print("logits:", tuple(y.shape))
