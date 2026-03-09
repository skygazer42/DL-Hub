
import torch
from torch import nn

from dlhub.pointcloud.ops import index_points, knn_indices

from ._common import check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "pointmlp_tiny": {"width": 64, "depth": 2, "k": 8},
    "pointmlp_small": {"width": 96, "depth": 3, "k": 16},
    "pointmlp_base": {"width": 128, "depth": 4, "k": 24},
}


class _LocalMix(nn.Module):
    def __init__(self, width: int, *, k: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.k = int(k)
        self.fc = nn.Sequential(
            nn.Linear(int(width) * 2, int(width)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(int(width), int(width)),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # xyz: (B,N,3), feat: (B,N,W)
        idx = knn_indices(xyz.to(torch.float32), self.k, exclude_self=True)
        neigh = index_points(feat, idx).mean(dim=2)
        return self.fc(torch.cat([feat, neigh], dim=-1))


class PointMLPSeg(nn.Module):
    """PointMLP semantic segmentation (toy): local mixing by neighbor mean."""

    def __init__(
        self, *, in_channels: int, num_classes: int, width: int, depth: int, k: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([_LocalMix(w, k=int(k), dropout=float(dropout)) for _ in range(int(depth))])
        self.cls = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, int(num_classes)))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))
        for blk in self.blocks:
            h = h + blk(xyz, h)
        return self.cls(h)


def build_pointmlp_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointmlp_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointMLPSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        k=int(cfg["k"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointmlp_segmenter3d(in_channels=3, num_classes=6, variant="pointmlp_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    (y.mean()).backward()
    print("logits:", tuple(y.shape))

