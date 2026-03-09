
import torch
from torch import nn

from ._common import PrototypeMaskHead, TransformerPointEncoder


_VARIANTS: dict[str, dict[str, object]] = {
    "maskformer3d_tiny": {"d_model": 64, "depth": 2, "prototypes": 16},
    "maskformer3d_small": {"d_model": 96, "depth": 3, "prototypes": 24},
    "maskformer3d_base": {"d_model": 128, "depth": 4, "prototypes": 32},
}


class MaskFormer3D(nn.Module):
    """MaskFormer3D (toy): transformer encoder + prototype mask head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        d_model: int,
        depth: int,
        num_prototypes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = TransformerPointEncoder(int(in_channels), int(d_model), depth=int(depth), dropout=float(dropout))
        self.head = PrototypeMaskHead(int(d_model), int(num_classes), num_prototypes=int(num_prototypes), dropout=float(dropout))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_maskformer3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "maskformer3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return MaskFormer3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        depth=int(cfg["depth"]),
        num_prototypes=int(cfg["prototypes"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_maskformer3d_instance_segmenter3d(in_channels=3, num_classes=6, variant="maskformer3d_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

