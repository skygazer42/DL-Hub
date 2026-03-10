from dataclasses import dataclass

import torch
from torch import nn

from .cnn import LayerNorm2d


def _make_divisible(v: int, divisor: int = 8) -> int:
    d = int(divisor)
    if d <= 0:
        raise ValueError("divisor must be > 0")
    x = int(v)
    if x <= 0:
        return d
    return int((x + d - 1) // d * d)


def _scale_dim(dim: int, width_mult: float, *, min_dim: int = 32, divisor: int = 8) -> int:
    d = max(int(min_dim), int(round(int(dim) * float(width_mult))))
    return _make_divisible(d, int(divisor))


def _parse_patch_variant(variant: str, *, default_patch_size: int) -> tuple[str, int]:
    name = str(variant).lower().strip()
    patch_size = int(default_patch_size)

    # Support names like `gmlp_tiny_p16`.
    if "_p" in name:
        base, suffix = name.rsplit("_p", 1)
        if suffix.isdigit():
            name = base
            patch_size = int(suffix)
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    return name, patch_size


# ---------------------------------------------------------------------------
# PoolFormer (pooling as token mixer)
# ---------------------------------------------------------------------------


class PoolingTokenMixer(nn.Module):
    def __init__(self, *, kernel_size: int = 3) -> None:
        super().__init__()
        k = int(kernel_size)
        if k <= 0:
            raise ValueError("kernel_size must be > 0")
        self.pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):
    def __init__(self, dim: int, *, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(int(dim))
        self.mixer = PoolingTokenMixer(kernel_size=3)
        self.drop1 = nn.Dropout(p=float(dropout))

        self.norm2 = LayerNorm2d(int(dim))
        hidden = max(8, int(round(int(dim) * float(mlp_ratio))))
        self.mlp = nn.Sequential(
            nn.Conv2d(int(dim), hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Conv2d(hidden, int(dim), kernel_size=1, bias=True),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.mixer(self.norm1(x)))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


@dataclass(frozen=True)
class PoolFormerConfig:
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    depth: int
    mlp_ratio: float
    dropout: float
    num_classes: int


class PoolFormerClassifier(nn.Module):
    def __init__(self, cfg: PoolFormerConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if int(cfg.depth) <= 0:
            raise ValueError("depth must be > 0")

        self.cfg = cfg
        self.patch_embed = nn.Conv2d(
            int(cfg.in_channels),
            int(cfg.embed_dim),
            kernel_size=int(cfg.patch_size),
            stride=int(cfg.patch_size),
            padding=0,
        )
        self.blocks = nn.Sequential(
            *[
                PoolFormerBlock(
                    int(cfg.embed_dim),
                    mlp_ratio=float(cfg.mlp_ratio),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(int(cfg.embed_dim))
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        x = self.drop(x)
        return self.head(x)


def build_poolformer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name, patch_size = _parse_patch_variant(variant, default_patch_size=4)

    if name in {"poolformer_tiny", "tiny", "poolformer"}:
        embed_dim, depth = 64, 6
    elif name in {"poolformer_small", "small"}:
        embed_dim, depth = 96, 8
    elif name in {"poolformer_base", "base"}:
        embed_dim, depth = 128, 12
    else:
        raise ValueError(
            "Unknown PoolFormer variant. Supported: poolformer_tiny|poolformer_small|poolformer_base (+ _p*)"
        )

    return PoolFormerClassifier(
        PoolFormerConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=_scale_dim(int(embed_dim), float(width_mult), min_dim=32, divisor=8),
            depth=int(depth),
            mlp_ratio=4.0,
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    )


# ---------------------------------------------------------------------------
# gMLP (gated MLP with spatial gating)
# ---------------------------------------------------------------------------


class SpatialGatingUnit(nn.Module):
    def __init__(self, *, num_tokens: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.proj = nn.Linear(int(num_tokens), int(num_tokens))
        self.drop = nn.Dropout(p=float(dropout))

        # Initialize close to identity gating for stability:
        # proj(v) ~= 1 => u * 1.
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2D) -> u, v: (B, T, D)
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(1, 2)).transpose(1, 2)
        v = self.drop(v)
        return u * v


class GMLPBlock(nn.Module):
    def __init__(self, dim: int, *, num_tokens: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        hidden = max(8, int(round(d * int(ff_mult))))
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 2 * hidden)
        self.act = nn.GELU()
        self.sgu = SpatialGatingUnit(num_tokens=int(num_tokens), dim=hidden, dropout=float(dropout))
        self.fc2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.sgu(y)
        y = self.fc2(y)
        return x + self.drop(y)


@dataclass(frozen=True)
class GMLPConfig:
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    depth: int
    ff_mult: int
    dropout: float
    num_classes: int


class GMLPClassifier(nn.Module):
    def __init__(self, cfg: GMLPConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if int(cfg.depth) <= 0:
            raise ValueError("depth must be > 0")

        self.cfg = cfg
        grid = int(cfg.image_size) // int(cfg.patch_size)
        num_tokens = grid * grid

        self.patch_embed = nn.Conv2d(
            int(cfg.in_channels),
            int(cfg.embed_dim),
            kernel_size=int(cfg.patch_size),
            stride=int(cfg.patch_size),
            padding=0,
        )
        self.blocks = nn.Sequential(
            *[
                GMLPBlock(
                    int(cfg.embed_dim),
                    num_tokens=int(num_tokens),
                    ff_mult=int(cfg.ff_mult),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)  # (B, C, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B, T, C)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def build_gmlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name, patch_size = _parse_patch_variant(variant, default_patch_size=8)

    if name in {"gmlp_tiny", "tiny", "gmlp"}:
        embed_dim, depth = 128, 6
    elif name in {"gmlp_small", "small"}:
        embed_dim, depth = 192, 8
    elif name in {"gmlp_base", "base"}:
        embed_dim, depth = 256, 12
    else:
        raise ValueError("Unknown gMLP variant. Supported: gmlp_tiny|gmlp_small|gmlp_base (+ _p*)")

    return GMLPClassifier(
        GMLPConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=_scale_dim(int(embed_dim), float(width_mult), min_dim=64, divisor=8),
            depth=int(depth),
            ff_mult=4,
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    )


# ---------------------------------------------------------------------------
# ResMLP (token-mixing linear + channel MLP)
# ---------------------------------------------------------------------------


class ResMLPBlock(nn.Module):
    def __init__(self, *, num_tokens: int, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        t = int(num_tokens)
        if t <= 0:
            raise ValueError("num_tokens must be > 0")

        self.ln1 = nn.LayerNorm(d)
        self.token_mixer = nn.Linear(t, t)
        self.drop1 = nn.Dropout(p=float(dropout))

        self.ln2 = nn.LayerNorm(d)
        hidden = max(8, int(round(d * float(mlp_ratio))))
        self.channel_mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, d),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y = self.ln1(x).transpose(1, 2)  # (B, C, T)
        y = self.token_mixer(y).transpose(1, 2)  # (B, T, C)
        x = x + self.drop1(y)

        y = self.channel_mlp(self.ln2(x))
        x = x + self.drop2(y)
        return x


@dataclass(frozen=True)
class ResMLPConfig:
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    depth: int
    mlp_ratio: float
    dropout: float
    num_classes: int


class ResMLPClassifier(nn.Module):
    def __init__(self, cfg: ResMLPConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if int(cfg.depth) <= 0:
            raise ValueError("depth must be > 0")

        self.cfg = cfg
        grid = int(cfg.image_size) // int(cfg.patch_size)
        num_tokens = grid * grid

        self.patch_embed = nn.Conv2d(
            int(cfg.in_channels),
            int(cfg.embed_dim),
            kernel_size=int(cfg.patch_size),
            stride=int(cfg.patch_size),
            padding=0,
        )
        self.blocks = nn.Sequential(
            *[
                ResMLPBlock(
                    num_tokens=int(num_tokens),
                    dim=int(cfg.embed_dim),
                    mlp_ratio=float(cfg.mlp_ratio),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)  # (B, C, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B, T, C)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def build_resmlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name, patch_size = _parse_patch_variant(variant, default_patch_size=8)

    if name in {"resmlp_tiny", "tiny", "resmlp"}:
        embed_dim, depth = 128, 6
    elif name in {"resmlp_small", "small"}:
        embed_dim, depth = 192, 8
    elif name in {"resmlp_base", "base"}:
        embed_dim, depth = 256, 12
    else:
        raise ValueError(
            "Unknown ResMLP variant. Supported: resmlp_tiny|resmlp_small|resmlp_base (+ _p*)"
        )

    return ResMLPClassifier(
        ResMLPConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=_scale_dim(int(embed_dim), float(width_mult), min_dim=64, divisor=8),
            depth=int(depth),
            mlp_ratio=4.0,
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    )


__all__ = [
    "build_gmlp_classifier",
    "build_poolformer_classifier",
    "build_resmlp_classifier",
]
