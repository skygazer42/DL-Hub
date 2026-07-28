import torch
from torch import nn

from ._common import MLPPointEncoder, SimilarityPivotHead

_VARIANTS: dict[str, dict[str, object]] = {
    "sgpn_tiny": {"width": 64, "depth": 2, "instances": 16},
    "sgpn_small": {"width": 96, "depth": 3, "instances": 24},
    "sgpn_base": {"width": 128, "depth": 4, "instances": 32},
}


class SGPN(nn.Module):
    """SGPN (compact): point embeddings; masks by similarity to pivots."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        num_instances: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = MLPPointEncoder(
            int(in_channels), int(width), depth=int(depth), dropout=float(dropout)
        )
        self.head = SimilarityPivotHead(
            int(width), int(num_classes), num_instances=int(num_instances), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_sgpn_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "sgpn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return SGPN(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        num_instances=int(cfg["instances"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_sgpn_instance_segmenter3d(in_channels=3, num_classes=6, variant="sgpn_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
