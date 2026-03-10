import torch
from torch import nn

from dlhub.pointcloud.ops import index_points, knn_indices

from ._common import check_points, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "pointsift_tiny": {"width": 64, "k": (8, 16, 24)},
    "pointsift_small": {"width": 96, "k": (8, 16, 32)},
    "pointsift_base": {"width": 128, "k": (16, 32, 48)},
}


class _ScaleMix(nn.Module):
    def __init__(self, width: int, k: int) -> None:
        super().__init__()
        self.k = int(k)
        w = int(width)
        self.fc = nn.Sequential(
            nn.Linear(w * 2, w), nn.ReLU(inplace=True), nn.Linear(w, w), nn.ReLU(inplace=True)
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(xyz.to(torch.float32), self.k, exclude_self=True)
        neigh = index_points(feat, idx).mean(dim=2)
        return self.fc(torch.cat([feat, neigh], dim=-1))


class PointSIFTSeg(nn.Module):
    """PointSIFT semantic segmentation (toy): multi-scale neighbor mixing."""

    def __init__(
        self, *, in_channels: int, num_classes: int, width: int, ks: tuple[int, int, int]
    ) -> None:
        super().__init__()
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.s1 = _ScaleMix(w, int(ks[0]))
        self.s2 = _ScaleMix(w, int(ks[1]))
        self.s3 = _ScaleMix(w, int(ks[2]))
        self.fuse = nn.Sequential(nn.Linear(w * 3, w), nn.ReLU(inplace=True))
        self.cls = nn.Sequential(
            nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, int(num_classes))
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))
        y1 = self.s1(xyz, h)
        y2 = self.s2(xyz, h)
        y3 = self.s3(xyz, h)
        y = self.fuse(torch.cat([y1, y2, y3], dim=-1))
        return self.cls(y)


def build_pointsift_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointsift_small",
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointSIFTSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        ks=tuple(int(x) for x in cfg["k"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointsift_segmenter3d(in_channels=3, num_classes=6, variant="pointsift_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))
