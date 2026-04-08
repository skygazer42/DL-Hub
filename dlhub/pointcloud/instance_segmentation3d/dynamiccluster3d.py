import torch
from torch import nn

from ._common import EdgeConvEncoder, SimilarityPivotHead

_VARIANTS: dict[str, dict[str, object]] = {
    "dynamiccluster3d_tiny": {"width": 64, "depth": 2, "k": 8, "instances": 16},
    "dynamiccluster3d_small": {"width": 96, "depth": 3, "k": 16, "instances": 24},
    "dynamiccluster3d_base": {"width": 128, "depth": 4, "k": 24, "instances": 32},
}


class DynamicCluster3D(nn.Module):
    """DynamicCluster3D (toy): embedding similarity to pivots defines soft groups."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        k: int,
        num_instances: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = EdgeConvEncoder(
            int(in_channels), int(width), depth=int(depth), k=int(k), dropout=float(dropout)
        )
        self.head = SimilarityPivotHead(
            int(width), int(num_classes), num_instances=int(num_instances), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_dynamiccluster3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "dynamiccluster3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return DynamicCluster3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        k=int(cfg["k"]),
        num_instances=int(cfg["instances"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_dynamiccluster3d_instance_segmenter3d(in_channels=3, num_classes=6, variant="dynamiccluster3d_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
