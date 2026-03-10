import torch
from torch import nn

from ._common import GridSpec2D, Projection2DEncoder, PrototypeMaskHead

_VARIANTS: dict[str, dict[str, object]] = {
    "fcis3d_tiny": {"width": 64, "bev": 24, "prototypes": 16},
    "fcis3d_small": {"width": 96, "bev": 32, "prototypes": 24},
    "fcis3d_base": {"width": 128, "bev": 40, "prototypes": 32},
}


class FCIS3D(nn.Module):
    """FCIS3D (toy): position-sensitive features via xyz MLP + prototype masks."""

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
        self.pos = nn.Sequential(
            nn.Linear(3, int(width) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(int(width) // 2, int(width)),
        )
        self.head = PrototypeMaskHead(
            int(width), int(num_classes), num_prototypes=int(num_prototypes), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        feat = feat + 0.1 * self.pos(xyz.to(feat.dtype)).tanh()
        return self.head(xyz, feat)


def build_fcis3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "fcis3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return FCIS3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev=int(cfg["bev"]),
        num_prototypes=int(cfg["prototypes"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_fcis3d_instance_segmenter3d(in_channels=3, num_classes=6, variant="fcis3d_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
