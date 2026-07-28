import torch
from torch import nn

from ._common import PointNet2Encoder, PrototypeMaskHead

_VARIANTS: dict[str, dict[str, object]] = {
    "pointnet2_inst_tiny": {"width": 48, "instances": 16},
    "pointnet2_inst_small": {"width": 64, "instances": 24},
    "pointnet2_inst_base": {"width": 96, "instances": 32},
}


class PointNet2Inst(nn.Module):
    """PointNet++ instance segmentation (compact): hierarchical encoder + prototype head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        num_instances: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = PointNet2Encoder(int(in_channels), int(width), dropout=float(dropout))
        self.head = PrototypeMaskHead(
            int(width), int(num_classes), num_prototypes=int(num_instances), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_pointnet2_inst_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointnet2_inst_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointNet2Inst(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_instances=int(cfg["instances"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_pointnet2_inst_instance_segmenter3d(
        in_channels=3, num_classes=6, variant="pointnet2_inst_tiny"
    )
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
