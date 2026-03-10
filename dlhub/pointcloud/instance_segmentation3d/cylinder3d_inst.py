import torch
from torch import nn

from ._common import CylinderEncoder, PrototypeMaskHead

_VARIANTS: dict[str, dict[str, object]] = {
    "cylinder3d_inst_tiny": {"width": 64, "h": 24, "w": 64, "instances": 16},
    "cylinder3d_inst_small": {"width": 96, "h": 32, "w": 96, "instances": 24},
    "cylinder3d_inst_base": {"width": 128, "h": 40, "w": 128, "instances": 32},
}


class Cylinder3DInst(nn.Module):
    """Cylinder3D instance segmentation (toy): cylindrical projection features + prototype masks."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        h: int,
        w: int,
        num_instances: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = CylinderEncoder(
            int(in_channels), int(width), h=int(h), w=int(w), dropout=float(dropout)
        )
        self.head = PrototypeMaskHead(
            int(width), int(num_classes), num_prototypes=int(num_instances), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_cylinder3d_inst_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "cylinder3d_inst_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return Cylinder3DInst(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        h=int(cfg["h"]),
        w=int(cfg["w"]),
        num_instances=int(cfg["instances"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_cylinder3d_inst_instance_segmenter3d(
        in_channels=3, num_classes=6, variant="cylinder3d_inst_tiny"
    )
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
