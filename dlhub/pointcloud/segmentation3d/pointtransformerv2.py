import torch
from torch import nn

from ._common import TransformerSegBase

_VARIANTS: dict[str, dict[str, object]] = {
    "pointtransformerv2_tiny": {"d_model": 72, "depth": 3, "pos_feats": 16},
    "pointtransformerv2_small": {"d_model": 104, "depth": 4, "pos_feats": 20},
    "pointtransformerv2_base": {"d_model": 144, "depth": 5, "pos_feats": 24},
}


class PointTransformerV2Seg(nn.Module):
    """Point Transformer v2 semantic segmentation (toy)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        d_model: int,
        depth: int,
        pos_feats: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = TransformerSegBase(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            d_model=int(d_model),
            depth=int(depth),
            dropout=float(dropout),
            pos_feats=int(pos_feats),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def build_pointtransformerv2_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointtransformerv2_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    return PointTransformerV2Seg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=int(int(cfg["d_model"]) * float(width_mult)),
        depth=int(cfg["depth"]),
        pos_feats=int(cfg["pos_feats"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointtransformerv2_segmenter3d(
        in_channels=3, num_classes=6, variant="pointtransformerv2_tiny"
    )
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
