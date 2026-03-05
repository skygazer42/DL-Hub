from __future__ import annotations

import torch
from torch import nn

from dlhub.pointcloud.ops import index_points, knn_indices

from ._common import check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "shellnet_tiny": {"width": 64, "depth": 2, "k": 16, "shells": 3},
    "shellnet_small": {"width": 96, "depth": 3, "k": 16, "shells": 4},
    "shellnet_base": {"width": 128, "depth": 4, "k": 24, "shells": 5},
}


class _ShellAgg(nn.Module):
    def __init__(self, width: int, *, shells: int) -> None:
        super().__init__()
        self.shells = int(shells)
        w = int(width)
        self.gate = nn.Sequential(nn.Linear(1, w), nn.ReLU(inplace=True), nn.Linear(w, self.shells))
        self.out = nn.Sequential(nn.Linear(w * self.shells, w), nn.ReLU(inplace=True))

    def forward(self, dist: torch.Tensor, neigh_feat: torch.Tensor) -> torch.Tensor:
        # dist: (B,N,k), neigh_feat: (B,N,k,W)
        # Soft-assign neighbors to shells based on normalized distance.
        d = dist / (dist.mean(dim=2, keepdim=True).clamp_min(1e-6))
        logits = self.gate(d.unsqueeze(-1))
        w = logits.softmax(dim=-1)  # (B,N,k,S)
        shells: list[torch.Tensor] = []
        for s in range(self.shells):
            ws = w[..., s : s + 1]
            shells.append((neigh_feat * ws).sum(dim=2) / (ws.sum(dim=2).clamp_min(1e-6)))
        return self.out(torch.cat(shells, dim=-1))


class ShellNetSeg(nn.Module):
    """ShellNet semantic segmentation (toy): distance shells around each point."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, depth: int, k: int, shells: int) -> None:
        super().__init__()
        self.k = int(k)
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.agg = _ShellAgg(w, shells=int(shells))
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True)) for _ in range(int(depth))])
        self.cls = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, int(num_classes)))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))
        idx = knn_indices(xyz.to(torch.float32), self.k, exclude_self=True)
        neigh_xyz = index_points(xyz, idx)
        rel = neigh_xyz - xyz.unsqueeze(2)
        dist = (rel * rel).sum(dim=-1).sqrt()
        neigh = index_points(h, idx)
        agg = self.agg(dist, neigh)
        for blk in self.blocks:
            h = h + blk(agg)
        return self.cls(h)


def build_shellnet_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "shellnet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return ShellNetSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        k=int(cfg["k"]),
        shells=int(cfg["shells"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_shellnet_segmenter3d(in_channels=3, num_classes=6, variant="shellnet_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

