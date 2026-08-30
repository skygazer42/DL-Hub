import math
from dataclasses import dataclass

import torch
from torch import nn

from ..ops import farthest_point_sample, index_points, knn_indices, knn_query
from .utils import _c


class PointTransformerBlock(nn.Module):
    def __init__(self, dim: int, *, k: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.k = int(k)
        self.num_heads = h
        self.head_dim = int(d // h)
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.norm1 = nn.LayerNorm(d)
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p=float(dropout))

        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d * 4, d),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3) | feat: (B, N, C)
        b, n, c = feat.shape
        if xyz.shape[:2] != (b, n) or xyz.shape[-1] != 3:
            raise ValueError("xyz and feat shapes must align on (B, N)")

        y = self.norm1(feat)
        idx = knn_indices(xyz, k=int(self.k))  # (B, N, k)

        nbr_feat = index_points(y, idx)  # (B, N, k, C)
        nbr_xyz = index_points(xyz, idx)  # (B, N, k, 3)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)  # (B, N, k, 3)

        q = self.to_q(y).view(b, n, self.num_heads, self.head_dim)  # (B, N, H, D)
        k = self.to_k(nbr_feat).view(b, n, int(self.k), self.num_heads, self.head_dim)
        v = self.to_v(nbr_feat).view(b, n, int(self.k), self.num_heads, self.head_dim)

        pos = self.pos_mlp(rel).view(b, n, int(self.k), self.num_heads, self.head_dim)
        attn_logits = (q.unsqueeze(2) * (k + pos)).sum(dim=-1) * float(self.scale)  # (B, N, k, H)
        attn = torch.softmax(attn_logits, dim=2)

        out = (attn.unsqueeze(-1) * (v + pos)).sum(dim=2)  # (B, N, H, D)
        out = out.reshape(b, n, c)
        out = self.drop(self.out(out))
        feat = feat + out

        z = self.norm2(feat)
        feat = feat + self.drop(self.ff(z))
        return feat


@dataclass(frozen=True)
class PointTransformerConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    k: int
    embed_dim: int
    depth: int
    num_heads: int


class PointTransformerClassifier(nn.Module):
    def __init__(self, cfg: PointTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)

        self.stem = nn.Sequential(
            nn.Conv1d(c_in, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                PointTransformerBlock(
                    d, k=int(cfg.k), num_heads=int(cfg.num_heads), dropout=float(cfg.dropout)
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
        x = points.to(torch.float32).transpose(1, 2).contiguous()  # (B, C, N)
        x = self.stem(x).transpose(1, 2).contiguous()  # (B, N, D)
        for blk in self.blocks:
            x = blk(xyz, x)
        x = self.norm(x)
        pooled = x.max(dim=1).values
        return self.head(pooled)


def build_point_transformer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "point_transformer",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"point_transformer", "pt"}:
        embed_dim, depth, heads, k = 128, 3, 4, 16
    elif name in {"point_transformer_tiny", "pt_tiny"}:
        embed_dim, depth, heads, k = 96, 2, 4, 12
    elif name in {"point_transformer_small", "pt_small"}:
        embed_dim, depth, heads, k = 160, 4, 5, 16
    else:
        raise ValueError(
            "Unknown PointTransformer variant. Supported: point_transformer|point_transformer_tiny|point_transformer_small"
        )

    return PointTransformerClassifier(
        PointTransformerConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            k=int(k),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(heads),
        )
    )


@dataclass(frozen=True)
class PCTConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    num_heads: int


class PCTClassifier(nn.Module):
    """A small Point Cloud Transformer (PCT-style), simplified.

    Uses full self-attention over points (O(N^2)), intended for small N (synthetic datasets).
    """

    def __init__(self, cfg: PCTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.embed = nn.Sequential(
            nn.Conv1d(c_in, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(cfg.num_heads),
            dim_feedforward=int(d) * 4,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(cfg.depth),
            enable_nested_tensor=False,
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
        x = points.to(torch.float32).transpose(1, 2).contiguous()  # (B, C, N)
        x = self.embed(x).transpose(1, 2).contiguous()  # (B, N, D)
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x.max(dim=1).values
        return self.head(pooled)


def build_pct_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pct",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pct", "pct_tiny"}:
        embed_dim, depth, heads = 128, 2, 4
    elif name in {"pct_small"}:
        embed_dim, depth, heads = 192, 3, 6
    elif name in {"pct_base"}:
        embed_dim, depth, heads = 256, 4, 8
    else:
        raise ValueError("Unknown PCT variant. Supported: pct|pct_small|pct_base")

    return PCTClassifier(
        PCTConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(heads),
        )
    )


__all__ = ["build_pct_classifier", "build_point_transformer_classifier"]


class PatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int,
        num_patches: int,
        k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.num_patches = int(num_patches)
        self.k = int(k)

        self.point_embed = nn.Sequential(
            nn.Linear(int(in_channels), int(embed_dim)),
            nn.ReLU(inplace=True),
        )
        self.group_mlp = nn.Sequential(
            nn.Linear(int(embed_dim) + 3, int(embed_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )
        self.pos = nn.Sequential(
            nn.Linear(3, int(embed_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.in_channels):
            raise ValueError(
                f"Expected points shape (B, N, C={self.in_channels}), got {tuple(points.shape)}"
            )

        xyz = points[..., :3].to(torch.float32)
        b, n, _ = xyz.shape
        s = min(int(self.num_patches), int(n))
        if s <= 0:
            raise ValueError("num_patches must be > 0")

        feat = self.point_embed(points.to(torch.float32))  # (B, N, D)
        fps_idx = farthest_point_sample(xyz, s)  # (B, S)
        centers = index_points(xyz, fps_idx)  # (B, S, 3)

        idx = knn_query(min(int(self.k), int(n)), xyz, centers)  # (B, S, k)
        grouped_xyz = index_points(xyz, idx) - centers.unsqueeze(2)
        grouped_feat = index_points(feat, idx)
        x = torch.cat([grouped_feat, grouped_xyz], dim=-1)
        x = self.group_mlp(x)
        tokens = x.max(dim=2).values  # (B, S, D)
        tokens = tokens + self.pos(centers)
        return tokens, centers


@dataclass(frozen=True)
class PointPatchTransformerConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    num_heads: int
    num_patches: int
    k: int


class PointPatchTransformerClassifier(nn.Module):
    def __init__(self, cfg: PointPatchTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)

        self.patch = PatchEmbed(
            in_channels=int(cfg.in_channels),
            embed_dim=int(d),
            num_patches=int(cfg.num_patches),
            k=int(cfg.k),
            dropout=float(cfg.dropout),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, int(d)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d),
            nhead=int(cfg.num_heads),
            dim_feedforward=int(d) * 4,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(cfg.depth),
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(int(d))
        self.head = nn.Sequential(
            nn.Linear(int(d), int(d)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(int(d), int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        tokens, _centers = self.patch(points)  # (B, S, D)
        b = tokens.shape[0]
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.encoder(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


def build_pointbert_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointbert",
) -> nn.Module:
    n = int(num_points)
    if n <= 0:
        raise ValueError("num_points must be > 0")
    name = str(variant).lower().strip()
    if name in {"pointbert", "pointbert_tiny"}:
        embed_dim, depth, heads, patches, k = 192, 2, 6, min(16, n), 16
    elif name in {"pointbert_small"}:
        embed_dim, depth, heads, patches, k = 256, 3, 8, min(24, n), 20
    else:
        raise ValueError("Unknown PointBERT variant. Supported: pointbert|pointbert_small")

    return PointPatchTransformerClassifier(
        PointPatchTransformerConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(heads),
            num_patches=int(patches),
            k=int(k),
        )
    )


def build_pointmae_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointmae",
) -> nn.Module:
    n = int(num_points)
    if n <= 0:
        raise ValueError("num_points must be > 0")
    name = str(variant).lower().strip()
    if name in {"pointmae", "pointmae_tiny"}:
        embed_dim, depth, heads, patches, k = 192, 2, 6, min(16, n), 16
    elif name in {"pointmae_small"}:
        embed_dim, depth, heads, patches, k = 256, 3, 8, min(24, n), 20
    else:
        raise ValueError("Unknown PointMAE variant. Supported: pointmae|pointmae_small")

    return PointPatchTransformerClassifier(
        PointPatchTransformerConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(heads),
            num_patches=int(patches),
            k=int(k),
        )
    )


__all__ = [
    "build_pct_classifier",
    "build_point_transformer_classifier",
    "build_pointbert_classifier",
    "build_pointmae_classifier",
]
