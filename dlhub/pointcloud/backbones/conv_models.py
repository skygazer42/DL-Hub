from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..ops import index_points, knn_indices
from .utils import ConvBNAct2d, _c, global_max_pool


class PointConvBlock(nn.Module):
    """Continuous convolution (PointConv-style), simplified.

    Uses a small MLP on relative coordinates to produce per-neighbor weights.
    """

    def __init__(self, in_dim: int, out_dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        self.k = int(k)
        self.weight_mlp = nn.Sequential(
            nn.Linear(3, max(8, int(out_dim) // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, int(out_dim) // 2), int(out_dim)),
        )
        self.value = nn.Linear(int(in_dim), int(out_dim), bias=False)
        self.proj = nn.Sequential(
            nn.Linear(int(out_dim), int(out_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3), feat: (B, N, C)
        b, n, _ = xyz.shape
        if feat.shape[:2] != (b, n):
            raise ValueError("xyz and feat shapes must align on (B, N)")

        idx = knn_indices(xyz, k=int(self.k))
        nbr_xyz = index_points(xyz, idx)  # (B, N, k, 3)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)
        w = self.weight_mlp(rel)  # (B, N, k, out)
        nbr_feat = index_points(feat, idx)  # (B, N, k, C)
        v = self.value(nbr_feat)  # (B, N, k, out)
        out = (w * v).sum(dim=2) / float(self.k)
        return self.proj(out)


@dataclass(frozen=True)
class PointConvConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int


class PointConvClassifier(nn.Module):
    def __init__(self, cfg: PointConvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(
            nn.Linear(c_in, d),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([PointConvBlock(d, d, k=int(cfg.k), dropout=float(cfg.dropout)) for _ in range(int(cfg.depth))])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(d, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}")
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))  # (B, N, D)
        for blk in self.blocks:
            feat = feat + blk(xyz, feat)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_pointconv_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointconv",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pointconv", "pointconv_tiny"}:
        embed_dim, depth, k = 128, 2, 16
    elif name in {"pointconv_small"}:
        embed_dim, depth, k = 160, 3, 16
    elif name in {"pointconv_base"}:
        embed_dim, depth, k = 192, 4, 20
    else:
        raise ValueError("Unknown PointConv variant. Supported: pointconv|pointconv_small|pointconv_base")

    return PointConvClassifier(
        PointConvConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
        )
    )


@dataclass(frozen=True)
class PointCNNConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    k: int


class PointCNNClassifier(nn.Module):
    """PointCNN-like classifier, simplified.

    Uses kNN grouping + per-neighborhood MLP + max pooling.
    """

    def __init__(self, cfg: PointCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(
            nn.Conv1d(c_in, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )
        self.local = nn.Sequential(
            ConvBNAct2d(d + 3, d, act="relu", dropout=float(cfg.dropout)),
            ConvBNAct2d(d, d, act="relu", dropout=float(cfg.dropout)),
        )
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(d, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}")

        xyz = points[..., :3].to(torch.float32)
        x = points.to(torch.float32).transpose(1, 2).contiguous()
        x = self.embed(x).transpose(1, 2).contiguous()  # (B, N, D)

        idx = knn_indices(xyz, k=int(self.cfg.k))
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)
        nbr_feat = index_points(x, idx)
        local_in = torch.cat([nbr_feat, rel], dim=-1)  # (B, N, k, D+3)
        local_in = local_in.permute(0, 3, 1, 2).contiguous()
        y = self.local(local_in)  # (B, D, N, k)
        y = torch.max(y, dim=-1).values  # (B, D, N)

        pooled = global_max_pool(y)
        return self.head(pooled)


def build_pointcnn_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointcnn",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pointcnn", "pointcnn_tiny"}:
        embed_dim, k = 128, 16
    elif name in {"pointcnn_small"}:
        embed_dim, k = 160, 20
    elif name in {"pointcnn_base"}:
        embed_dim, k = 192, 24
    else:
        raise ValueError("Unknown PointCNN variant. Supported: pointcnn|pointcnn_small|pointcnn_base")

    return PointCNNClassifier(
        PointCNNConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            k=int(k),
        )
    )


class KPConvBlock(nn.Module):
    """KPConv-like block, simplified.

    Uses a small set of learnable kernel centers and weights points by RBF distance to centers.
    """

    def __init__(self, in_dim: int, out_dim: int, *, k: int, num_kpoints: int, dropout: float) -> None:
        super().__init__()
        self.k = int(k)
        self.num_kpoints = int(num_kpoints)
        self.centers = nn.Parameter(torch.randn(self.num_kpoints, 3) * 0.2)
        self.radius = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.value = nn.Linear(int(in_dim), int(out_dim), bias=False)
        self.mix = nn.Linear(int(out_dim), int(out_dim), bias=False)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(xyz, k=int(self.k))
        nbr_xyz = index_points(xyz, idx)  # (B, N, k, 3)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)  # (B, N, k, 3)

        # (B, N, k, P)
        dist2 = ((rel.unsqueeze(3) - self.centers.view(1, 1, 1, self.num_kpoints, 3)) ** 2).sum(dim=-1)
        sigma2 = (self.radius.abs() + 1e-6) ** 2
        w = torch.exp(-dist2 / (2.0 * sigma2))  # RBF
        w = w / (w.sum(dim=3, keepdim=True) + 1e-6)

        nbr_feat = index_points(feat, idx)  # (B, N, k, C)
        v = self.value(nbr_feat)  # (B, N, k, out)
        # Mix across kernel points.
        v_k = (w.unsqueeze(-1) * v.unsqueeze(3)).sum(dim=2)  # (B, N, P, out)
        out = v_k.mean(dim=2)
        out = self.drop(self.mix(out))
        return out


@dataclass(frozen=True)
class KPConvConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int
    num_kpoints: int


class KPConvClassifier(nn.Module):
    def __init__(self, cfg: KPConvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [
                KPConvBlock(d, d, k=int(cfg.k), num_kpoints=int(cfg.num_kpoints), dropout=float(cfg.dropout))
                for _ in range(int(cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(d, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}")
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))
        for blk in self.blocks:
            feat = feat + blk(xyz, feat)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_kpconv_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "kpconv",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"kpconv", "kpconv_tiny"}:
        embed_dim, depth, k, p = 128, 2, 16, 8
    elif name in {"kpconv_small"}:
        embed_dim, depth, k, p = 160, 3, 16, 12
    elif name in {"kpconv_base"}:
        embed_dim, depth, k, p = 192, 4, 20, 16
    else:
        raise ValueError("Unknown KPConv variant. Supported: kpconv|kpconv_small|kpconv_base")

    return KPConvClassifier(
        KPConvConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
            num_kpoints=int(p),
        )
    )


__all__ = [
    "build_kpconv_classifier",
    "build_pointcnn_classifier",
    "build_pointconv_classifier",
    "build_shellnet_classifier",
]


@dataclass(frozen=True)
class ShellNetConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    num_shells: int
    depth: int


class ShellNetClassifier(nn.Module):
    """ShellNet-style classifier (radial shells), simplified.

    This implementation:
    - Sorts points by radius (||xyz||)
    - Splits into `num_shells` shells
    - Pools per-shell features
    - Runs a small 1D conv over shells
    """

    def __init__(self, cfg: ShellNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(
            nn.Conv1d(c_in, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )

        layers: list[nn.Module] = []
        for _ in range(int(cfg.depth)):
            layers.extend(
                [
                    nn.Conv1d(d, d, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(d),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=float(cfg.dropout)),
                ]
            )
        self.shell_mixer = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(d, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}")

        xyz = points[..., :3].to(torch.float32)  # (B, N, 3)
        b, n, _ = xyz.shape
        s = int(self.cfg.num_shells)
        if s <= 0:
            raise ValueError("num_shells must be > 0")
        s = min(s, n)

        x = points.to(torch.float32).transpose(1, 2).contiguous()  # (B, C, N)
        x = self.embed(x).transpose(1, 2).contiguous()  # (B, N, D)

        r = xyz.norm(dim=-1)  # (B, N)
        order = torch.argsort(r, dim=1)  # (B, N)
        x_sorted = index_points(x, order)  # (B, N, D)

        chunks = torch.chunk(x_sorted, s, dim=1)
        shell_feat = torch.stack([c.max(dim=1).values for c in chunks], dim=2)  # (B, D, S)

        shell_feat = self.shell_mixer(shell_feat)
        pooled = shell_feat.max(dim=-1).values  # (B, D)
        return self.head(pooled)


def build_shellnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "shellnet",
) -> nn.Module:
    n = int(num_points)
    if n <= 0:
        raise ValueError("num_points must be > 0")
    name = str(variant).lower().strip()
    if name in {"shellnet", "shellnet_tiny"}:
        embed_dim, shells, depth = 128, 8, 2
    elif name in {"shellnet_small"}:
        embed_dim, shells, depth = 160, 12, 3
    elif name in {"shellnet_base"}:
        embed_dim, shells, depth = 192, 16, 4
    else:
        raise ValueError("Unknown ShellNet variant. Supported: shellnet|shellnet_small|shellnet_base")

    shells = min(int(shells), int(n))
    return ShellNetClassifier(
        ShellNetConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            num_shells=int(shells),
            depth=int(depth),
        )
    )
