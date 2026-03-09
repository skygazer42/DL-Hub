
import torch
from torch import nn

from dlhub.pointcloud.ops import index_points, knn_indices

from ._common import check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "pointmixer_tiny": {"width": 64, "depth": 2, "k": 8},
    "pointmixer_small": {"width": 96, "depth": 3, "k": 16},
    "pointmixer_base": {"width": 128, "depth": 4, "k": 24},
}


class _MixerBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        w = int(width)
        self.channel = nn.Sequential(nn.Linear(w, w * 2), nn.ReLU(inplace=True), nn.Linear(w * 2, w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.channel(x).tanh()


class PointMixerSeg(nn.Module):
    """PointMixer semantic segmentation (toy): token mix via kNN mean + channel mix MLP."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, depth: int, k: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.k = int(k)
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.mix = nn.ModuleList([_MixerBlock(w) for _ in range(int(depth))])
        self.cls = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Dropout(float(dropout)), nn.Linear(w, int(num_classes)))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))

        idx = knn_indices(xyz.to(torch.float32), self.k, exclude_self=True)
        neigh = index_points(h, idx).mean(dim=2)
        h = h + 0.1 * neigh
        for blk in self.mix:
            h = blk(h)
        return self.cls(h)


def build_pointmixer_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointmixer_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointMixerSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        k=int(cfg["k"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointmixer_segmenter3d(in_channels=3, num_classes=6, variant="pointmixer_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

