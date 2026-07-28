import torch
from torch import nn

from ._common import GridSpec3D, Voxel3DSegBase

_VARIANTS: dict[str, dict[str, object]] = {
    "voxelunet_tiny": {"width": 32, "grid": (6, 24, 24)},
    "voxelunet_small": {"width": 48, "grid": (8, 32, 32)},
    "voxelunet_base": {"width": 64, "grid": (10, 40, 40)},
}


class VoxelUNetSeg(nn.Module):
    """Voxel UNet semantic segmentation (compact): dense 3D UNet on a coarse voxel grid."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        grid: tuple[int, int, int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d, h, w = (int(x) for x in grid)
        self.net = Voxel3DSegBase(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            grid=GridSpec3D(d=d, h=h, w=w),
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def build_voxelunet_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "voxelunet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return VoxelUNetSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        grid=tuple(cfg["grid"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_voxelunet_segmenter3d(in_channels=3, num_classes=6, variant="voxelunet_tiny")
    x = torch.randn(2, 256, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
