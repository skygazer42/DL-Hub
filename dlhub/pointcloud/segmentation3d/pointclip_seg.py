import torch
from torch import nn

from ._common import TransformerSegBase

_VARIANTS: dict[str, dict[str, object]] = {
    "pointclip_seg_tiny": {"d_model": 64, "depth": 3},
    "pointclip_seg_small": {"d_model": 96, "depth": 4},
    "pointclip_seg_base": {"d_model": 128, "depth": 6},
}


class PointclipSeg(nn.Module):
    """PointMAE semantic segmentation (toy): transformer encoder + light head."""

    def __init__(
        self, *, in_channels: int, num_classes: int, d_model: int, depth: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.net = TransformerSegBase(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            d_model=int(d_model),
            depth=int(depth),
            dropout=float(dropout),
            pos_feats=16,
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def build_pointclip_seg_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointclip_seg_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return PointclipSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointclip_seg_segmenter3d(in_channels=3, num_classes=6, variant="pointclip_seg_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

