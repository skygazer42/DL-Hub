import torch
from torch import nn

from ._common import GridSpec2D, Projection2DEncoder, PrototypeMaskHead

_VARIANTS: dict[str, dict[str, object]] = {
    "sparseinst3d_tiny": {"width": 48, "bev": 24, "prototypes": 16},
    "sparseinst3d_small": {"width": 64, "bev": 32, "prototypes": 24},
    "sparseinst3d_base": {"width": 96, "bev": 40, "prototypes": 32},
}


class SparseInst3D(nn.Module):
    """SparseInst3D (compact): BEV projection + sparse prototypes."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev: int,
        num_prototypes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        grid = GridSpec2D(h=int(bev), w=int(bev))
        self.enc = Projection2DEncoder(
            int(in_channels), int(width), grid=grid, dropout=float(dropout)
        )
        self.head = PrototypeMaskHead(
            int(width), int(num_classes), num_prototypes=int(num_prototypes), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_sparseinst3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "sparseinst3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return SparseInst3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev=int(cfg["bev"]),
        num_prototypes=int(cfg["prototypes"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_sparseinst3d_instance_segmenter3d(
        in_channels=3, num_classes=6, variant="sparseinst3d_tiny"
    )
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
