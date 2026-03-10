from dataclasses import dataclass

import torch
from torch import nn

from ..ops import farthest_point_sample, index_points, knn_indices, knn_query
from .utils import _c


class SpiderConvBlock(nn.Module):
    """SpiderCNN-style local conv, simplified.

    Uses a polynomial basis over relative xyz to modulate neighbor features.
    """

    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.basis = nn.Sequential(
            nn.Linear(10, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, d),
        )
        self.value = nn.Linear(d, d, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

    @staticmethod
    def _poly_basis(rel: torch.Tensor) -> torch.Tensor:
        x, y, z = rel.unbind(dim=-1)
        return torch.stack(
            [
                torch.ones_like(x),
                x,
                y,
                z,
                x * x,
                y * y,
                z * z,
                x * y,
                y * z,
                z * x,
            ],
            dim=-1,
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3) | feat: (B, N, D)
        idx = knn_indices(xyz, k=int(self.k))
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)
        basis = self._poly_basis(rel)  # (B, N, k, 10)
        w = self.basis(basis)  # (B, N, k, D)

        nbr_feat = index_points(feat, idx)  # (B, N, k, D)
        v = self.value(nbr_feat)
        out = (w * v).mean(dim=2)  # (B, N, D)
        return self.proj(out)


@dataclass(frozen=True)
class SpiderCNNConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int


class SpiderCNNClassifier(nn.Module):
    def __init__(self, cfg: SpiderCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [
                SpiderConvBlock(d, k=int(cfg.k), dropout=float(cfg.dropout))
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
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))
        for blk in self.blocks:
            feat = feat + blk(xyz, feat)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_spidercnn_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "spidercnn",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"spidercnn", "spidercnn_tiny"}:
        embed_dim, depth, k = 128, 2, 16
    elif name in {"spidercnn_small"}:
        embed_dim, depth, k = 160, 3, 20
    else:
        raise ValueError("Unknown SpiderCNN variant. Supported: spidercnn|spidercnn_small")

    return SpiderCNNClassifier(
        SpiderCNNConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
        )
    )


class RelationShapeConvBlock(nn.Module):
    """RS-CNN-style local relation conv, simplified."""

    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.rel = nn.Sequential(
            nn.Linear(d * 2 + 4, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        self.value = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(xyz, k=int(self.k))
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)  # (B, N, k, 3)
        dist = rel.norm(dim=-1, keepdim=True)  # (B, N, k, 1)

        nbr_feat = index_points(feat, idx)
        c = feat.unsqueeze(2).expand_as(nbr_feat)
        gate = self.rel(torch.cat([c, nbr_feat, rel, dist], dim=-1))
        v = self.value(nbr_feat)
        out = (gate * v).mean(dim=2)
        return self.proj(out)


@dataclass(frozen=True)
class RSCNNConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int


class RSCNNClassifier(nn.Module):
    def __init__(self, cfg: RSCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [
                RelationShapeConvBlock(d, k=int(cfg.k), dropout=float(cfg.dropout))
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
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))
        for blk in self.blocks:
            feat = feat + blk(xyz, feat)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_rscnn_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "rscnn",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"rscnn", "rscnn_tiny"}:
        embed_dim, depth, k = 128, 2, 16
    elif name in {"rscnn_small"}:
        embed_dim, depth, k = 160, 3, 20
    else:
        raise ValueError("Unknown RS-CNN variant. Supported: rscnn|rscnn_small")

    return RSCNNClassifier(
        RSCNNConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
        )
    )


class PAConvBlock(nn.Module):
    """PAConv-style dynamic kernel aggregation, simplified."""

    def __init__(self, dim: int, *, k: int, num_kernels: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.m = int(num_kernels)
        self.bank = nn.Parameter(torch.randn(self.m, d, d) * (1.0 / (d**0.5)))
        self.coef = nn.Sequential(
            nn.Linear(3, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, self.m),
        )
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(xyz, k=int(self.k))
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)  # (B, N, k, 3)
        nbr_feat = index_points(feat, idx)  # (B, N, k, D)

        coef = torch.softmax(self.coef(rel), dim=-1)  # (B, N, k, M)
        # Apply kernel bank: (B,N,k,D) x (M,D,D) -> (B,N,k,M,D)
        v = torch.einsum("bnkd,mdh->bnkmh", nbr_feat, self.bank)
        out = (coef.unsqueeze(-1) * v).sum(dim=3)  # (B, N, k, D)
        out = out.mean(dim=2)
        return self.proj(out)


@dataclass(frozen=True)
class PAConvConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int
    num_kernels: int


class PAConvClassifier(nn.Module):
    def __init__(self, cfg: PAConvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [
                PAConvBlock(
                    d, k=int(cfg.k), num_kernels=int(cfg.num_kernels), dropout=float(cfg.dropout)
                )
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
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))
        for blk in self.blocks:
            feat = feat + blk(xyz, feat)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_paconv_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "paconv",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"paconv", "paconv_tiny"}:
        embed_dim, depth, k, m = 128, 2, 16, 4
    elif name in {"paconv_small"}:
        embed_dim, depth, k, m = 160, 3, 20, 6
    else:
        raise ValueError("Unknown PAConv variant. Supported: paconv|paconv_small")

    return PAConvClassifier(
        PAConvConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
            num_kernels=int(m),
        )
    )


@dataclass(frozen=True)
class Point2SeqConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    k: int


class Point2SeqClassifier(nn.Module):
    """Point2Sequence-style neighborhood sequence aggregation, simplified."""

    def __init__(self, cfg: Point2SeqConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True))
        self.gru = nn.GRU(input_size=d, hidden_size=d, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
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
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))  # (B, N, D)
        idx = knn_indices(xyz, k=int(self.cfg.k))
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)
        dist = rel.norm(dim=-1)  # (B, N, k)
        order = torch.argsort(dist, dim=2)
        idx_sorted = torch.gather(idx, 2, order)
        seq = index_points(feat, idx_sorted)  # (B, N, k, D)

        b, n, k, d = seq.shape
        seq = seq.view(b * n, k, d)
        out, _ = self.gru(seq)
        last = out[:, -1, :].view(b, n, d)
        feat = feat + self.proj(last)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_point2seq_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "point2seq",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"point2seq", "point2sequence"}:
        embed_dim, k = 128, 16
    elif name in {"point2seq_small"}:
        embed_dim, k = 160, 20
    else:
        raise ValueError("Unknown Point2Seq variant. Supported: point2seq|point2seq_small")

    return Point2SeqClassifier(
        Point2SeqConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            k=int(k),
        )
    )


class AttnPool(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, dropout: float) -> None:
        super().__init__()
        d_in = int(in_dim)
        d_out = int(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d_out, d_out),
        )
        self.attn = nn.Sequential(
            nn.Linear(d_out, max(8, d_out // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, d_out // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, K, C)
        y = self.mlp(x)
        w = self.attn(y).squeeze(-1)  # (B, S, K)
        w = torch.softmax(w, dim=2)
        return (w.unsqueeze(-1) * y).sum(dim=2)


@dataclass(frozen=True)
class ASNLConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    npoint1: int
    k1: int
    npoint2: int
    k2: int


class ASNLClassifier(nn.Module):
    """ASNL-style sampling + attention pooling, simplified."""

    def __init__(self, cfg: ASNLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d1 = _c(64, float(cfg.width_mult), min_ch=16, divisor=8)
        d2 = _c(128, float(cfg.width_mult), min_ch=32, divisor=8)
        d3 = _c(256, float(cfg.width_mult), min_ch=64, divisor=8)

        self.pool1 = AttnPool(in_dim=3, out_dim=d1, dropout=cfg.dropout)
        self.pool2 = AttnPool(in_dim=3 + d1, out_dim=d2, dropout=cfg.dropout)
        self.proj = nn.Sequential(nn.Linear(d2, d3), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(d3, d3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(d3, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)

        fps1 = farthest_point_sample(xyz, int(self.cfg.npoint1))
        new_xyz1 = index_points(xyz, fps1)
        idx1 = knn_query(int(self.cfg.k1), xyz, new_xyz1)
        grouped1 = index_points(xyz, idx1) - new_xyz1.unsqueeze(2)  # (B, S1, k1, 3)
        feat1 = self.pool1(grouped1)  # (B, S1, d1)

        fps2 = farthest_point_sample(new_xyz1, int(self.cfg.npoint2))
        new_xyz2 = index_points(new_xyz1, fps2)
        idx2 = knn_query(int(self.cfg.k2), new_xyz1, new_xyz2)
        grouped2_xyz = index_points(new_xyz1, idx2) - new_xyz2.unsqueeze(2)
        grouped2_feat = index_points(feat1, idx2)
        grouped2 = torch.cat([grouped2_xyz, grouped2_feat], dim=-1)
        feat2 = self.pool2(grouped2)  # (B, S2, d2)

        feat = self.proj(feat2)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_asnl_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "asnl",
) -> nn.Module:
    n = int(num_points)
    if n <= 0:
        raise ValueError("num_points must be > 0")
    _ = str(variant)
    npoint1 = max(16, n // 2)
    npoint2 = max(8, n // 8)
    return ASNLClassifier(
        ASNLConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            npoint1=int(npoint1),
            k1=16,
            npoint2=int(npoint2),
            k2=12,
        )
    )


@dataclass(frozen=True)
class RandLAConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    k: int


class RandLANetClassifier(nn.Module):
    """RandLA-Net-style local attention aggregation with random sampling, simplified."""

    def __init__(self, cfg: RandLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True))
        self.attn = nn.Sequential(
            nn.Linear(d * 2 + 3, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(d, 1),
        )
        self.proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(d, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))

        idx = knn_indices(xyz, k=int(self.cfg.k))
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)
        nbr_feat = index_points(feat, idx)
        c = feat.unsqueeze(2).expand_as(nbr_feat)
        w = self.attn(torch.cat([c, nbr_feat, rel], dim=-1)).squeeze(-1)
        w = torch.softmax(w, dim=2)
        agg = (w.unsqueeze(-1) * nbr_feat).sum(dim=2)
        feat = self.norm(feat + self.proj(agg))

        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_randlanet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "randlanet",
) -> nn.Module:
    _ = int(num_points)
    _ = str(variant)
    return RandLANetClassifier(
        RandLAConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=128,
            k=16,
        )
    )


@dataclass(frozen=True)
class PVCNNConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    grid_size: int


class PVCNNClassifier(nn.Module):
    """Point-Voxel CNN (PVCNN-style), simplified.

    - Point branch: per-point MLP + global max
    - Voxel branch: voxelize point features -> small 3D CNN -> global avg
    - Fuse: concat + classifier head
    """

    def __init__(self, cfg: PVCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        g = int(cfg.grid_size)
        if g <= 0:
            raise ValueError("grid_size must be > 0")
        self.grid_size = g

        self.point_embed = nn.Sequential(
            nn.Linear(c_in, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
        )

        self.voxel_cnn = nn.Sequential(
            nn.Conv3d(d, d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(d),
            nn.ReLU(inplace=True),
            nn.Conv3d(d, d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(d),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(d, int(cfg.num_classes)),
        )

    def _voxelize(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3) in roughly [-1, 1], feat: (B, N, D)
        b, n, d = feat.shape
        g = int(self.grid_size)
        coords = (xyz.clamp(-1.0, 1.0) + 1.0) * 0.5  # [0,1]
        ijk = (coords * float(g - 1)).round().to(torch.long)  # (B, N, 3)
        ijk = ijk.clamp(0, g - 1)
        idx = ijk[..., 0] * (g * g) + ijk[..., 1] * g + ijk[..., 2]  # (B, N)

        base = torch.arange(b, device=feat.device).view(b, 1) * (g * g * g)
        idx_global = (idx + base).reshape(-1)  # (B*N,)
        feat_flat = feat.reshape(-1, d)

        vox = torch.zeros((b * g * g * g, d), device=feat.device, dtype=feat.dtype)
        vox.index_add_(0, idx_global, feat_flat)
        counts = torch.zeros((b * g * g * g, 1), device=feat.device, dtype=feat.dtype)
        ones = torch.ones((b * n, 1), device=feat.device, dtype=feat.dtype)
        counts.index_add_(0, idx_global, ones)
        vox = vox / counts.clamp(min=1.0)

        vox = vox.view(b, g * g * g, d).transpose(1, 2).contiguous().view(b, d, g, g, g)
        return vox

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.point_embed(points.to(torch.float32))  # (B, N, D)

        point_global = feat.max(dim=1).values
        vox = self._voxelize(xyz, feat)
        vox = self.voxel_cnn(vox)
        voxel_global = vox.mean(dim=(2, 3, 4))

        fused = torch.cat([point_global, voxel_global], dim=-1)
        return self.head(fused)


def build_pvcnn_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pvcnn",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pvcnn", "pvcnn_tiny"}:
        embed_dim, grid = 128, 16
    elif name in {"pvcnn_small"}:
        embed_dim, grid = 160, 20
    else:
        raise ValueError("Unknown PVCNN variant. Supported: pvcnn|pvcnn_small")

    return PVCNNClassifier(
        PVCNNConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            grid_size=int(grid),
        )
    )


@dataclass(frozen=True)
class SimpleViewConfig:
    in_channels: int
    num_classes: int
    dropout: float
    grid_size: int


class SimpleViewClassifier(nn.Module):
    """SimpleView-style multi-view projection baseline, simplified."""

    def __init__(self, cfg: SimpleViewConfig) -> None:
        super().__init__()
        self.cfg = cfg
        g = int(cfg.grid_size)
        if g <= 0:
            raise ValueError("grid_size must be > 0")
        self.grid_size = g

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(128, int(cfg.num_classes)),
        )

    def _project(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: (B, N) in [-1,1]
        bsz, n = a.shape
        g = int(self.grid_size)
        u = ((a.clamp(-1.0, 1.0) + 1.0) * 0.5 * float(g - 1)).round().to(torch.long).clamp(0, g - 1)
        v = ((b.clamp(-1.0, 1.0) + 1.0) * 0.5 * float(g - 1)).round().to(torch.long).clamp(0, g - 1)
        idx = u * g + v  # (B, N)

        base = torch.arange(bsz, device=a.device).view(bsz, 1) * (g * g)
        idx_global = (idx + base).reshape(-1)
        img = torch.zeros((bsz * g * g,), device=a.device, dtype=torch.float32)
        ones = torch.ones((bsz * n,), device=a.device, dtype=torch.float32)
        img.index_add_(0, idx_global, ones)
        img = img.view(bsz, 1, g, g)
        img = img / (img.amax(dim=(2, 3), keepdim=True).clamp(min=1.0))
        return img

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)  # (B, N, 3)
        x, y, z = xyz.unbind(dim=-1)
        img_xy = self._project(x, y)
        img_yz = self._project(y, z)
        img_zx = self._project(z, x)
        img = torch.cat([img_xy, img_yz, img_zx], dim=1)  # (B, 3, H, W)

        feat = self.cnn(img)
        pooled = feat.mean(dim=(2, 3))
        return self.head(pooled)


def build_simpleview_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "simpleview",
) -> nn.Module:
    _ = float(width_mult)
    _ = int(num_points)
    _ = str(variant)
    return SimpleViewClassifier(
        SimpleViewConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            dropout=float(dropout),
            grid_size=32,
        )
    )


class CurveConvBlock(nn.Module):
    """CurveNet-style curve aggregation, simplified.

    Sorts kNN neighbors by distance and applies a 1D conv over the neighbor sequence.
    """

    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(xyz, k=int(self.k))
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)
        dist = rel.norm(dim=-1)  # (B, N, k)
        order = torch.argsort(dist, dim=2)
        idx_sorted = torch.gather(idx, 2, order)
        seq = index_points(feat, idx_sorted)  # (B, N, k, D)

        b, n, k, d = seq.shape
        seq = seq.view(b * n, k, d).transpose(1, 2).contiguous()  # (B*N, D, k)
        seq = self.conv(seq)
        pooled = seq.max(dim=-1).values.view(b, n, d)
        return self.proj(pooled)


@dataclass(frozen=True)
class CurveNetConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int


class CurveNetClassifier(nn.Module):
    def __init__(self, cfg: CurveNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [
                CurveConvBlock(d, k=int(cfg.k), dropout=float(cfg.dropout))
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
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))
        for blk in self.blocks:
            feat = feat + blk(xyz, feat)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_curvenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "curvenet",
) -> nn.Module:
    _ = int(num_points)
    _ = str(variant)
    return CurveNetClassifier(
        CurveNetConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=160,
            depth=3,
            k=20,
        )
    )


@dataclass(frozen=True)
class GDANetConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int


class GDABlock(nn.Module):
    """Geometry-discriminative attention block, simplified."""

    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.rel = nn.Sequential(
            nn.Linear(3, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, d),
        )
        self.score = nn.Sequential(
            nn.Linear(d, max(8, d // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, d // 2), 1),
        )
        self.proj = nn.Linear(d, d, bias=False)
        self.se = nn.Sequential(
            nn.Linear(d, max(8, d // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, d // 4), d),
            nn.Sigmoid(),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(xyz, k=int(self.k))
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)
        rel_e = self.rel(rel)

        nbr_feat = index_points(feat, idx)
        x = nbr_feat + rel_e
        w = self.score(x).squeeze(-1)
        w = torch.softmax(w, dim=2)
        agg = (w.unsqueeze(-1) * x).sum(dim=2)
        agg = self.proj(agg)

        # Global channel reweighting.
        g = feat.mean(dim=1)
        gate = self.se(g).unsqueeze(1)
        return feat + agg * gate


class GDANetClassifier(nn.Module):
    def __init__(self, cfg: GDANetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [GDABlock(d, k=int(cfg.k), dropout=float(cfg.dropout)) for _ in range(int(cfg.depth))]
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
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))
        for blk in self.blocks:
            feat = blk(xyz, feat)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_gdanet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "gdanet",
) -> nn.Module:
    _ = int(num_points)
    _ = str(variant)
    return GDANetClassifier(
        GDANetConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=160,
            depth=3,
            k=20,
        )
    )


class PointSIFTBlock(nn.Module):
    """PointSIFT-style octant aggregation, simplified."""

    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.mlp = nn.Sequential(
            nn.Linear(d * 8, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, d),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(xyz, k=int(self.k))
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)  # (B, N, k, 3)
        nbr_feat = index_points(feat, idx)  # (B, N, k, D)

        sign = (rel > 0).to(torch.int64)
        octant = sign[..., 0] * 4 + sign[..., 1] * 2 + sign[..., 2]  # (B, N, k) in [0,7]

        outs: list[torch.Tensor] = []
        for o in range(8):
            mask = (octant == o).unsqueeze(-1)  # (B, N, k, 1)
            masked = nbr_feat.masked_fill(~mask, float("-inf"))
            pooled = masked.max(dim=2).values
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            outs.append(pooled)

        x = torch.cat(outs, dim=-1)  # (B, N, 8D)
        return self.mlp(x)


@dataclass(frozen=True)
class PointSIFTConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int


class PointSIFTClassifier(nn.Module):
    def __init__(self, cfg: PointSIFTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [
                PointSIFTBlock(d, k=int(cfg.k), dropout=float(cfg.dropout))
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
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))
        for blk in self.blocks:
            feat = feat + blk(xyz, feat)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_pointsift_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointsift",
) -> nn.Module:
    _ = int(num_points)
    _ = str(variant)
    return PointSIFTClassifier(
        PointSIFTConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=160,
            depth=2,
            k=32,
        )
    )


__all__ = [
    "build_asnl_classifier",
    "build_curvenet_classifier",
    "build_gdanet_classifier",
    "build_paconv_classifier",
    "build_point2seq_classifier",
    "build_pointsift_classifier",
    "build_pvcnn_classifier",
    "build_randlanet_classifier",
    "build_rscnn_classifier",
    "build_simpleview_classifier",
    "build_spidercnn_classifier",
]
