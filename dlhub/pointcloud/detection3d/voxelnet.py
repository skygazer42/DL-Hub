
import torch
from torch import nn

from ._common import BEVBoxSpec, BEVAnchorFreeDetector3D


_VARIANTS: dict[str, dict[str, object]] = {
    "voxelnet_tiny": {"width": 48, "bev_h": 24, "bev_w": 24, "topk": 48},
    "voxelnet_small": {"width": 64, "bev_h": 32, "bev_w": 32, "topk": 64},
    "voxelnet_base": {"width": 96, "bev_h": 40, "bev_w": 40, "topk": 96},
}


class VoxelNet(nn.Module):
    """VoxelNet (toy): points -> BEV pseudo-voxels -> dense head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev_h: int,
        bev_w: int,
        topk: int,
        with_yaw: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.det = BEVAnchorFreeDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            bev=BEVBoxSpec(h=int(bev_h), w=int(bev_w)),
            topk=int(topk),
            with_yaw=bool(with_yaw),
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.det(points)


def build_voxelnet_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "voxelnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return VoxelNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        topk=int(cfg["topk"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_voxelnet_detector3d(in_channels=3, num_classes=4, variant="voxelnet_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    loss = out["boxes"].mean() + out["cls_logits"].mean()
    loss.backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

