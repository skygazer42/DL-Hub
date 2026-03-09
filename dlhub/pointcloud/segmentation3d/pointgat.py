
import torch
from torch import nn

from dlhub.pointcloud.ops import index_points, knn_indices

from ._common import check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "pointgat_tiny": {"width": 64, "depth": 2, "k": 8},
    "pointgat_small": {"width": 96, "depth": 3, "k": 16},
    "pointgat_base": {"width": 128, "depth": 4, "k": 24},
}


class _GraphAttn(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        w = int(width)
        self.q = nn.Linear(w, w)
        self.k = nn.Linear(w, w)
        self.v = nn.Linear(w, w)
        self.out = nn.Linear(w, w)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # x: (B,N,W), idx: (B,N,k)
        neigh = index_points(x, idx)  # (B,N,k,W)
        q = self.q(x).unsqueeze(2)  # (B,N,1,W)
        k = self.k(neigh)
        v = self.v(neigh)
        attn = (q * k).sum(dim=-1, keepdim=True) / (q.shape[-1] ** 0.5)
        w = attn.softmax(dim=2)
        agg = (v * w).sum(dim=2)  # (B,N,W)
        return torch.relu(self.out(agg))


class PointGATSeg(nn.Module):
    """PointGAT semantic segmentation (toy): kNN graph attention layers."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, depth: int, k: int) -> None:
        super().__init__()
        self.k = int(k)
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([_GraphAttn(w) for _ in range(int(depth))])
        self.cls = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, int(num_classes)))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))
        idx = knn_indices(xyz.to(torch.float32), self.k, exclude_self=True)
        for blk in self.blocks:
            h = h + blk(h, idx)
        return self.cls(h)


def build_pointgat_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointgat_small",
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointGATSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        k=int(cfg["k"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointgat_segmenter3d(in_channels=3, num_classes=6, variant="pointgat_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
