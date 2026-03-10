import torch
from torch import nn

from ._common import GridSpec3D, PointVoxelFusionEncoder, PrototypeMaskHead

_VARIANTS: dict[str, dict[str, object]] = {
    "pvcnn_inst_tiny": {"width": 64, "grid": (6, 24, 24), "instances": 16},
    "pvcnn_inst_small": {"width": 96, "grid": (8, 32, 32), "instances": 24},
    "pvcnn_inst_base": {"width": 128, "grid": (10, 40, 40), "instances": 32},
}


class PVCNNInst(nn.Module):
    """PVCNN instance segmentation (toy): point-voxel fusion + prototype head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        grid: tuple[int, int, int],
        num_instances: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d, h, w = (int(x) for x in grid)
        self.enc = PointVoxelFusionEncoder(
            int(in_channels), int(width), grid=GridSpec3D(d=d, h=h, w=w), dropout=float(dropout)
        )
        self.head = PrototypeMaskHead(
            int(width), int(num_classes), num_prototypes=int(num_instances), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_pvcnn_inst_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pvcnn_inst_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PVCNNInst(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        grid=tuple(cfg["grid"]),
        num_instances=int(cfg["instances"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_pvcnn_inst_instance_segmenter3d(
        in_channels=3, num_classes=6, variant="pvcnn_inst_tiny"
    )
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
