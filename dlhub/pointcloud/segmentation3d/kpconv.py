import torch
from torch import nn
from torch.nn import functional as F

from dlhub.pointcloud.ops import index_points, knn_indices

from ._common import check_points, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "kpconv_tiny": {"width": 48, "k": 16, "kernels": 8},
    "kpconv_small": {"width": 64, "k": 16, "kernels": 12},
    "kpconv_base": {"width": 96, "k": 24, "kernels": 16},
}


class _KPConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, *, k: int, num_kernels: int, sigma: float = 0.5
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.num_kernels = int(num_kernels)
        self.sigma = float(sigma)
        self.kernel_points = nn.Parameter(torch.randn(self.num_kernels, 3) * 0.5)
        self.lin = nn.Linear(int(in_channels), int(out_channels))
        self.out = nn.Linear(int(out_channels) * self.num_kernels, int(out_channels))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(xyz.to(torch.float32), self.k, exclude_self=True)  # (B,N,k)
        neigh_xyz = index_points(xyz, idx)  # (B,N,k,3)
        neigh_feat = index_points(feat, idx)  # (B,N,k,C)

        rel = neigh_xyz - xyz.unsqueeze(2)  # (B,N,k,3)
        kp = self.kernel_points.view(1, 1, 1, self.num_kernels, 3)  # (1,1,1,K,3)
        rel2 = rel.unsqueeze(3) - kp  # (B,N,k,K,3)
        dist2 = (rel2 * rel2).sum(dim=-1)  # (B,N,k,K)
        w = torch.exp(-0.5 * dist2 / (self.sigma * self.sigma))  # (B,N,k,K)
        w = w / (w.sum(dim=2, keepdim=True) + 1e-6)

        v = self.lin(neigh_feat)  # (B,N,k,D)
        # Weighted sum over neighbors for each kernel point
        out = (v.unsqueeze(3) * w.unsqueeze(-1)).sum(dim=2)  # (B,N,K,D)
        out = out.reshape(out.shape[0], out.shape[1], self.num_kernels * out.shape[-1])
        return F.relu(self.out(out), inplace=True)


class KPConvSeg(nn.Module):
    """KPConv semantic segmentation (toy): KPConv blocks + per-point logits."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        k: int,
        kernels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.kp1 = _KPConv(w, w, k=int(k), num_kernels=int(kernels))
        self.kp2 = _KPConv(w, w, k=int(k), num_kernels=int(kernels))
        self.cls = nn.Sequential(
            nn.Linear(w, w),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(w, int(num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))
        h = h + self.kp1(xyz, h)
        h = h + self.kp2(xyz, h)
        return self.cls(h)


def build_kpconv_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "kpconv_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return KPConvSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        k=int(cfg["k"]),
        kernels=int(cfg["kernels"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_kpconv_segmenter3d(in_channels=3, num_classes=6, variant="kpconv_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    (y.mean()).backward()
    print("logits:", tuple(y.shape))
