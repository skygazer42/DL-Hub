
import torch
from torch import nn

from dlhub.pointcloud.ops import index_points, knn_indices

from ._common import check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "paconv_tiny": {"width": 64, "depth": 2, "k": 8},
    "paconv_small": {"width": 96, "depth": 3, "k": 16},
    "paconv_base": {"width": 128, "depth": 4, "k": 24},
}


class _PAConv(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        w = int(width)
        self.pos = nn.Sequential(nn.Linear(3, w // 2), nn.ReLU(inplace=True), nn.Linear(w // 2, w))
        self.feat = nn.Linear(w, w)
        self.out = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True))

    def forward(self, rel: torch.Tensor, neigh_feat: torch.Tensor) -> torch.Tensor:
        # rel: (B,N,k,3), neigh_feat: (B,N,k,W)
        w = torch.tanh(self.pos(rel.to(torch.float32)))
        v = self.feat(neigh_feat)
        return self.out((v * (1.0 + w)).mean(dim=2))


class PAConvSeg(nn.Module):
    """PAConv semantic segmentation (toy): position-adaptive neighbor mixing."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, depth: int, k: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.k = int(k)
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([_PAConv(w) for _ in range(int(depth))])
        self.cls = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Dropout(float(dropout)), nn.Linear(w, int(num_classes)))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))
        idx = knn_indices(xyz.to(torch.float32), self.k, exclude_self=True)
        neigh_xyz = index_points(xyz, idx)
        rel = neigh_xyz - xyz.unsqueeze(2)
        for blk in self.blocks:
            neigh = index_points(h, idx)
            h = h + blk(rel, neigh)
        return self.cls(h)


def build_paconv_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "paconv_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PAConvSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        k=int(cfg["k"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_paconv_segmenter3d(in_channels=3, num_classes=6, variant="paconv_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

