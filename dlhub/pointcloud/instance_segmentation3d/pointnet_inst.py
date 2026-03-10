import torch
from torch import nn

from ._common import MLPPointEncoder, PrototypeMaskHead

_VARIANTS: dict[str, dict[str, object]] = {
    "pointnet_inst_tiny": {"width": 48, "depth": 2, "prototypes": 16},
    "pointnet_inst_small": {"width": 64, "depth": 3, "prototypes": 24},
    "pointnet_inst_base": {"width": 96, "depth": 4, "prototypes": 32},
}


class PointNetInst(nn.Module):
    """PointNet instance segmentation (toy): Point MLP encoder + prototype masks."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        num_prototypes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = MLPPointEncoder(
            int(in_channels), int(width), depth=int(depth), dropout=float(dropout)
        )
        self.head = PrototypeMaskHead(
            int(width), int(num_classes), num_prototypes=int(num_prototypes), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_pointnet_inst_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointnet_inst_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointNetInst(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        num_prototypes=int(cfg["prototypes"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_pointnet_inst_instance_segmenter3d(
        in_channels=3, num_classes=6, variant="pointnet_inst_tiny"
    )
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
