from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..ops import index_points, knn_indices
from .utils import ConvBNAct2d, _c, global_max_pool


class LocalMLPAggregation(nn.Module):
    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.mlp = nn.Sequential(
            ConvBNAct2d(d * 2 + 3, d, act="relu", dropout=float(dropout)),
            ConvBNAct2d(d, d, act="relu", dropout=float(dropout)),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3) | feat: (B, D, N)
        b, d, n = feat.shape
        if xyz.shape[:2] != (b, n) or xyz.shape[-1] != 3:
            raise ValueError("xyz and feat shapes must align")

        idx = knn_indices(xyz, k=int(self.k))  # (B, N, k)
        feat_n = index_points(feat.transpose(1, 2).contiguous(), idx)  # (B, N, k, D)
        feat_c = feat.transpose(1, 2).contiguous().unsqueeze(2).expand(-1, -1, int(self.k), -1)
        xyz_n = index_points(xyz, idx)
        rel = (xyz_n - xyz.unsqueeze(2)).to(torch.float32)

        x = torch.cat([feat_c, feat_n - feat_c, rel], dim=-1)  # (B, N, k, 2D+3)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 2D+3, N, k)
        x = self.mlp(x)  # (B, D, N, k)
        x = torch.max(x, dim=-1).values  # (B, D, N)
        x = self.proj(x)
        return self.act(feat + x)


@dataclass(frozen=True)
class PointMLPConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int


class PointMLPClassifier(nn.Module):
    def __init__(self, cfg: PointMLPConfig) -> None:
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
            [LocalMLPAggregation(d, k=int(cfg.k), dropout=float(cfg.dropout)) for _ in range(int(cfg.depth))]
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
        x = points.to(torch.float32).transpose(1, 2).contiguous()  # (B, C, N)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(xyz, x)
        pooled = global_max_pool(x)
        return self.head(pooled)


def build_pointmlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointmlp",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pointmlp", "pointmlp_tiny"}:
        embed_dim, depth, k = 128, 3, 16
    elif name in {"pointmlp_small"}:
        embed_dim, depth, k = 160, 4, 16
    elif name in {"pointmlp_base"}:
        embed_dim, depth, k = 192, 5, 20
    else:
        raise ValueError("Unknown PointMLP variant. Supported: pointmlp|pointmlp_small|pointmlp_base")

    return PointMLPClassifier(
        PointMLPConfig(
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
class PointNeXtConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int


class PointNeXtClassifier(nn.Module):
    """A PointNeXt-like classifier (local MLP aggregation + stage depth), simplified."""

    def __init__(self, cfg: PointNeXtConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        for _ in range(int(cfg.depth)):
            blocks.append(LocalMLPAggregation(d, k=int(cfg.k), dropout=float(cfg.dropout)))
        self.blocks = nn.Sequential(*blocks)
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
        x = self.stem(x)
        # Sequential LocalMLPAggregation expects xyz each time.
        for blk in self.blocks:
            x = blk(xyz, x)
        pooled = global_max_pool(x)
        return self.head(pooled)


def build_pointnext_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str,
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pointnext_tiny", "pointnext"}:
        embed_dim, depth, k = 128, 4, 16
    elif name in {"pointnext_small"}:
        embed_dim, depth, k = 160, 6, 16
    elif name in {"pointnext_base"}:
        embed_dim, depth, k = 192, 8, 20
    else:
        raise ValueError("Unknown PointNeXt variant. Supported: pointnext_tiny|pointnext_small|pointnext_base")

    return PointNeXtClassifier(
        PointNeXtConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
        )
    )


__all__ = [
    "build_pointmixer_classifier",
    "build_pointmlp_classifier",
    "build_pointnext_classifier",
]


class ChannelMix(nn.Module):
    def __init__(self, dim: int, *, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d * 4, d),
        )
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, N)
        y = x.transpose(1, 2).contiguous()  # (B, N, D)
        y = y + self.drop(self.ff(self.norm(y)))
        return y.transpose(1, 2).contiguous()


class PointMixerBlock(nn.Module):
    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        self.local = LocalMLPAggregation(int(dim), k=int(k), dropout=float(dropout))
        self.channel = ChannelMix(int(dim), dropout=float(dropout))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        feat = self.local(xyz, feat)
        feat = self.channel(feat)
        return feat


@dataclass(frozen=True)
class PointMixerConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int


class PointMixerClassifier(nn.Module):
    def __init__(self, cfg: PointMixerConfig) -> None:
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
            [PointMixerBlock(d, k=int(cfg.k), dropout=float(cfg.dropout)) for _ in range(int(cfg.depth))]
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
        x = points.to(torch.float32).transpose(1, 2).contiguous()  # (B, C, N)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(xyz, x)
        pooled = global_max_pool(x)
        return self.head(pooled)


def build_pointmixer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointmixer",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pointmixer", "pointmixer_tiny"}:
        embed_dim, depth, k = 128, 3, 16
    elif name in {"pointmixer_small"}:
        embed_dim, depth, k = 160, 4, 20
    else:
        raise ValueError("Unknown PointMixer variant. Supported: pointmixer|pointmixer_small")

    return PointMixerClassifier(
        PointMixerConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
        )
    )
