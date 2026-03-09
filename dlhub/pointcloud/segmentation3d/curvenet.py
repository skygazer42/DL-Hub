
import torch
from torch import nn

from dlhub.pointcloud.ops import index_points, knn_indices

from ._common import check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "curvenet_tiny": {"width": 64, "k": 16},
    "curvenet_small": {"width": 96, "k": 24},
    "curvenet_base": {"width": 128, "k": 32},
}


class CurveNetSeg(nn.Module):
    """CurveNet semantic segmentation (toy): ordered neighbor conv along distance."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, k: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.k = int(k)
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv1d(w, w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv1d(w, w, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, int(num_classes)))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))

        idx = knn_indices(xyz.to(torch.float32), self.k, exclude_self=True)  # (B,N,k)
        neigh_xyz = index_points(xyz, idx)
        rel = neigh_xyz - xyz.unsqueeze(2)
        dist = (rel * rel).sum(dim=-1)  # (B,N,k)
        perm = dist.argsort(dim=-1)
        idx_sorted = idx.gather(-1, perm)
        neigh_h = index_points(h, idx_sorted)  # (B,N,k,W)

        b, n, k, w = neigh_h.shape
        z = neigh_h.reshape(b * n, k, w).transpose(1, 2).contiguous()  # (B*N,W,k)
        z = self.conv(z).max(dim=-1).values  # (B*N,W)
        z = z.view(b, n, w)
        return self.cls(h + z)


def build_curvenet_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "curvenet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return CurveNetSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        k=int(cfg["k"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_curvenet_segmenter3d(in_channels=3, num_classes=6, variant="curvenet_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

