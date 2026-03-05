from __future__ import annotations

import torch
from torch import nn

from ._common import GridSpec3D, PrototypeMaskHead, Voxel3DEncoder


_VARIANTS: dict[str, dict[str, object]] = {
    "occuseg_tiny": {"width": 48, "grid": (6, 24, 24), "prototypes": 16},
    "occuseg_small": {"width": 64, "grid": (8, 32, 32), "prototypes": 24},
    "occuseg_base": {"width": 96, "grid": (10, 40, 40), "prototypes": 32},
}


class OccuSeg(nn.Module):
    """OccuSeg (toy): voxelized features + prototype instance masks."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        grid: tuple[int, int, int],
        num_prototypes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d, h, w = (int(x) for x in grid)
        self.enc = Voxel3DEncoder(int(in_channels), int(width), grid=GridSpec3D(d=d, h=h, w=w), dropout=float(dropout))
        self.head = PrototypeMaskHead(int(width), int(num_classes), num_prototypes=int(num_prototypes), dropout=float(dropout))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_occuseg_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "occuseg_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return OccuSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        grid=tuple(cfg["grid"]),
        num_prototypes=int(cfg["prototypes"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_occuseg_instance_segmenter3d(in_channels=3, num_classes=6, variant="occuseg_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

