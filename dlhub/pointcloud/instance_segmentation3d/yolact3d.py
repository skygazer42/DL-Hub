from __future__ import annotations

import torch
from torch import nn

from ._common import MLPPointEncoder, YOLACTHead


_VARIANTS: dict[str, dict[str, object]] = {
    "yolact3d_tiny": {"width": 64, "instances": 16, "prototypes": 8},
    "yolact3d_small": {"width": 96, "instances": 24, "prototypes": 8},
    "yolact3d_base": {"width": 128, "instances": 32, "prototypes": 12},
}


class YOLACT3D(nn.Module):
    """YOLACT3D (toy): point features -> prototypes + coeffs."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        num_instances: int,
        num_prototypes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = MLPPointEncoder(int(in_channels), int(width), depth=3, dropout=float(dropout))
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


def build_yolact3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "yolact3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return YOLACT3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_instances=int(cfg["instances"]),
        num_prototypes=int(cfg["prototypes"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_yolact3d_instance_segmenter3d(in_channels=3, num_classes=6, variant="yolact3d_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

