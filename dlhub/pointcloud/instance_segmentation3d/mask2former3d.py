import torch
from torch import nn

from ._common import PointNet2Encoder, QueryMaskHead

_VARIANTS: dict[str, dict[str, object]] = {
    "mask2former3d_tiny": {"width": 48, "queries": 16},
    "mask2former3d_small": {"width": 64, "queries": 24},
    "mask2former3d_base": {"width": 96, "queries": 32},
}


class Mask2Former3D(nn.Module):
    """Mask2Former3D (toy): PointNet++ hierarchy + query mask head (multi-scale proxy)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = PointNet2Encoder(int(in_channels), int(width), dropout=float(dropout))
        self.head = QueryMaskHead(
            int(width), int(num_classes), num_queries=int(num_queries), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_mask2former3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "mask2former3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return Mask2Former3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_mask2former3d_instance_segmenter3d(
        in_channels=3, num_classes=6, variant="mask2former3d_tiny"
    )
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
