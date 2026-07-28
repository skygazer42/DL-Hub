import torch
from torch import nn

from ._common import EdgeConv, QueryHead, check_points, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "dgcnn_det_tiny": {"width": 64, "k": 8, "queries": 32},
    "dgcnn_det_small": {"width": 96, "k": 16, "queries": 48},
    "dgcnn_det_base": {"width": 128, "k": 24, "queries": 64},
}


class DGCNNDet(nn.Module):
    """DGCNN baseline detector (compact): EdgeConv tokens -> global -> query head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        k: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge1 = EdgeConv(int(in_channels), int(width), k=int(k), dropout=float(dropout))
        self.edge2 = EdgeConv(int(width), int(width), k=int(k), dropout=float(dropout))
        self.head = QueryHead(
            int(width), int(num_classes), num_queries=int(num_queries), with_yaw=True
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        t = self.edge2(self.edge1(x))
        context = t.mean(dim=1)
        return self.head(context)


def build_dgcnn_det_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "dgcnn_det_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return DGCNNDet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        k=int(cfg["k"]),
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_dgcnn_det_detector3d(in_channels=3, num_classes=3, variant="dgcnn_det_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
