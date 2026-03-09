
import torch
from torch import nn

from ._common import GridSpec2D, Projection2DEncoder, YOLACTHead


_VARIANTS: dict[str, dict[str, object]] = {
    "solov2_3d_tiny": {"width": 64, "bev": 24, "instances": 16, "prototypes": 8},
    "solov2_3d_small": {"width": 96, "bev": 32, "instances": 24, "prototypes": 8},
    "solov2_3d_base": {"width": 128, "bev": 40, "instances": 32, "prototypes": 12},
}


class SOLOv2_3D(nn.Module):
    """SOLOv2-3D (toy): BEV projection provides spatial inductive bias; masks via coeffs+prototypes."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev: int,
        num_instances: int,
        num_prototypes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        grid = GridSpec2D(h=int(bev), w=int(bev))
        self.enc = Projection2DEncoder(int(in_channels), int(width), grid=grid, dropout=float(dropout))
        self.head = YOLACTHead(
            int(width),
            int(num_classes),
            num_instances=int(num_instances),
            num_prototypes=int(num_prototypes),
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_solov2_3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "solov2_3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return SOLOv2_3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev=int(cfg["bev"]),
        num_instances=int(cfg["instances"]),
        num_prototypes=int(cfg["prototypes"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_solov2_3d_instance_segmenter3d(in_channels=3, num_classes=6, variant="solov2_3d_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

