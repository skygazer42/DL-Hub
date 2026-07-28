import torch
from torch import nn

from dlhub.pointcloud.ops import index_points, knn_indices

from ._common import check_points, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "pointgcn_tiny": {"width": 64, "depth": 2, "k": 8},
    "pointgcn_small": {"width": 96, "depth": 3, "k": 16},
    "pointgcn_base": {"width": 128, "depth": 4, "k": 24},
}


class _GraphConv(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        w = int(width)
        self.self_fc = nn.Linear(w, w)
        self.nei_fc = nn.Linear(w, w)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        neigh = index_points(x, idx).mean(dim=2)
        return self.act(self.self_fc(x) + self.nei_fc(neigh))


class PointGCNSeg(nn.Module):
    """PointGCN semantic segmentation (compact): kNN graph conv layers."""

    def __init__(
        self, *, in_channels: int, num_classes: int, width: int, depth: int, k: int
    ) -> None:
        super().__init__()
        self.k = int(k)
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([_GraphConv(w) for _ in range(int(depth))])
        self.cls = nn.Sequential(
            nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, int(num_classes))
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))
        idx = knn_indices(xyz.to(torch.float32), self.k, exclude_self=True)
        for blk in self.blocks:
            h = blk(h, idx)
        return self.cls(h)


def build_pointgcn_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointgcn_small",
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointGCNSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        k=int(cfg["k"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointgcn_segmenter3d(in_channels=3, num_classes=6, variant="pointgcn_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
