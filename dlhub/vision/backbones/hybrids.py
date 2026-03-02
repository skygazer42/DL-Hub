from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .cnn import SqueezeExcite
from .transformers import TransformerEncoderBlock


def _make_divisible(v: int, divisor: int = 8) -> int:
    d = int(divisor)
    if d <= 0:
        raise ValueError("divisor must be > 0")
    x = int(v)
    if x <= 0:
        return d
    return int((x + d - 1) // d * d)


def _c(ch: int, width_mult: float, *, min_ch: int = 8, divisor: int = 8) -> int:
    v = max(int(min_ch), int(round(int(ch) * float(width_mult))))
    return _make_divisible(v, int(divisor))


def _pick_heads(embed_dim: int, *, preferred: list[int]) -> int:
    d = int(embed_dim)
    for h in preferred:
        h = int(h)
        if h > 0 and d % h == 0:
            return h
    return 1


def _parse_patch_variant(variant: str, *, default_patch_size: int) -> tuple[str, int]:
    name = str(variant).lower().strip()
    patch_size = int(default_patch_size)
    if "_p" in name:
        base, suffix = name.rsplit("_p", 1)
        if suffix.isdigit():
            name = base
            patch_size = int(suffix)
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    return name, patch_size


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int | None = None,
        groups: int = 1,
        act: str = "silu",
    ) -> None:
        k = int(kernel_size)
        if padding is None:
            padding = k // 2

        act_name = str(act).lower().strip()
        if act_name in {"silu", "swish"}:
            act_layer: nn.Module = nn.SiLU(inplace=True)
        elif act_name in {"relu"}:
            act_layer = nn.ReLU(inplace=True)
        elif act_name in {"gelu"}:
            act_layer = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {act!r}")

        super().__init__(
            nn.Conv2d(
                int(in_ch),
                int(out_ch),
                kernel_size=int(k),
                stride=int(stride),
                padding=int(padding),
                groups=int(groups),
                bias=False,
            ),
            nn.BatchNorm2d(int(out_ch)),
            act_layer,
        )


class MBConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        s = int(stride)
        if s not in {1, 2}:
            raise ValueError("stride must be 1 or 2")
        hidden = int(in_ch) * int(expand_ratio)
        self.use_res = s == 1 and in_ch == out_ch

        layers: list[nn.Module] = []
        if hidden != in_ch:
            layers.append(ConvBNAct(in_ch, hidden, kernel_size=1, stride=1, padding=0, act="silu"))
        layers.append(ConvBNAct(hidden, hidden, kernel_size=3, stride=s, groups=hidden, act="silu"))
        if float(se_ratio) > 0:
            layers.append(SqueezeExcite(hidden, se_ratio=float(se_ratio)))
        layers.append(nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.net = nn.Sequential(*layers)
        self.drop = nn.Dropout2d(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        out = self.net(x)
        out = self.drop(out)
        if self.use_res:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# MobileViT (simplified)
# ---------------------------------------------------------------------------


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int,
        out_ch: int,
        transformer_dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        tdim = int(transformer_dim)
        heads = int(num_heads)
        depth = int(depth)
        if depth <= 0:
            raise ValueError("depth must be > 0")

        self.local = nn.Sequential(
            ConvBNAct(in_ch, in_ch, kernel_size=3, stride=1, groups=in_ch, act="silu"),
            ConvBNAct(in_ch, tdim, kernel_size=1, stride=1, padding=0, act="silu"),
        )
        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=tdim,
                    num_heads=heads,
                    ff_dim=tdim * 4,
                    dropout=float(dropout),
                )
                for _ in range(depth)
            ]
        )
        self.proj = ConvBNAct(tdim, out_ch, kernel_size=1, stride=1, padding=0, act="silu")
        self.fuse = ConvBNAct(in_ch + out_ch, out_ch, kernel_size=3, stride=1, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b, _, h, w = x.shape
        y = self.local(x)  # (B, tdim, H, W)
        y = y.flatten(2).transpose(1, 2)  # (B, T, tdim)
        y = self.transformer(y)
        y = y.transpose(1, 2).contiguous().view(b, -1, h, w)
        y = self.proj(y)
        return self.fuse(torch.cat([x, y], dim=1))


@dataclass(frozen=True)
class MobileViTConfig:
    image_size: int
    in_channels: int
    num_classes: int
    widths: tuple[int, int, int, int, int]
    transformer_dims: tuple[int, int]
    transformer_depths: tuple[int, int]
    dropout: float


class MobileViTClassifier(nn.Module):
    def __init__(self, cfg: MobileViTConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % 16 != 0:
            raise ValueError("MobileViTClassifier expects image_size divisible by 16")

        w1, w2, w3, w4, w5 = map(int, cfg.widths)
        t1, t2 = map(int, cfg.transformer_dims)
        d1, d2 = map(int, cfg.transformer_depths)

        self.stem = ConvBNAct(int(cfg.in_channels), w1, kernel_size=3, stride=2, act="silu")
        self.stage1 = MBConv(w1, w2, stride=1, expand_ratio=2, se_ratio=0.0, dropout=cfg.dropout)
        self.stage2 = MBConv(w2, w3, stride=2, expand_ratio=2, se_ratio=0.25, dropout=cfg.dropout)
        self.stage3 = nn.Sequential(
            MBConv(w3, w4, stride=2, expand_ratio=2, se_ratio=0.25, dropout=cfg.dropout),
            MobileViTBlock(
                in_ch=w4,
                out_ch=w4,
                transformer_dim=t1,
                num_heads=_pick_heads(t1, preferred=[8, 6, 4, 3, 2, 1]),
                depth=d1,
                dropout=cfg.dropout,
            ),
        )
        self.stage4 = nn.Sequential(
            MBConv(w4, w5, stride=2, expand_ratio=2, se_ratio=0.25, dropout=cfg.dropout),
            MobileViTBlock(
                in_ch=w5,
                out_ch=w5,
                transformer_dim=t2,
                num_heads=_pick_heads(t2, preferred=[8, 6, 4, 3, 2, 1]),
                depth=d2,
                dropout=cfg.dropout,
            ),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(w5, int(cfg.num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_mobilevit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    w = float(width_mult)

    if name in {"mobilevit_tiny", "tiny", "mobilevit"}:
        widths = (16, 24, 32, 64, 96)
        transformer_dims = (64, 96)
        transformer_depths = (2, 2)
    elif name in {"mobilevit_small", "small"}:
        widths = (16, 32, 48, 80, 128)
        transformer_dims = (96, 128)
        transformer_depths = (2, 4)
    elif name in {"mobilevit_base", "base"}:
        widths = (24, 48, 64, 128, 192)
        transformer_dims = (128, 192)
        transformer_depths = (4, 4)
    else:
        raise ValueError("Unknown MobileViT variant. Supported: mobilevit_tiny|mobilevit_small|mobilevit_base")

    widths_scaled = tuple(_c(int(c), w, min_ch=16, divisor=8) for c in widths)
    t_scaled = tuple(_c(int(c), w, min_ch=64, divisor=8) for c in transformer_dims)

    return MobileViTClassifier(
        MobileViTConfig(
            image_size=int(image_size),
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            widths=tuple(map(int, widths_scaled)),
            transformer_dims=tuple(map(int, t_scaled)),
            transformer_depths=tuple(map(int, transformer_depths)),
            dropout=float(dropout),
        )
    )


# ---------------------------------------------------------------------------
# CoAtNet (simplified conv + attention stages)
# ---------------------------------------------------------------------------


class Attention2DBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.block = TransformerEncoderBlock(
            embed_dim=int(dim),
            num_heads=int(num_heads),
            ff_dim=int(dim) * 4,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f"Expected channels={self.dim}, got {c}")
        t = x.flatten(2).transpose(1, 2)  # (B, T, C)
        t = self.block(t)
        return t.transpose(1, 2).contiguous().view(b, c, h, w)


@dataclass(frozen=True)
class CoAtNetConfig:
    image_size: int
    in_channels: int
    num_classes: int
    widths: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    dropout: float


class CoAtNetClassifier(nn.Module):
    def __init__(self, cfg: CoAtNetConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % 16 != 0:
            raise ValueError("CoAtNetClassifier expects image_size divisible by 16")

        w1, w2, w3, w4 = map(int, cfg.widths)
        d1, d2, d3, d4 = map(int, cfg.depths)

        self.stem = ConvBNAct(int(cfg.in_channels), w1, kernel_size=3, stride=2, act="silu")

        s1: list[nn.Module] = []
        for i in range(d1):
            s1.append(MBConv(w1, w1, stride=1, expand_ratio=2, se_ratio=0.25, dropout=cfg.dropout))
        self.stage1 = nn.Sequential(*s1)

        s2: list[nn.Module] = [MBConv(w1, w2, stride=2, expand_ratio=2, se_ratio=0.25, dropout=cfg.dropout)]
        for _ in range(1, d2):
            s2.append(MBConv(w2, w2, stride=1, expand_ratio=2, se_ratio=0.25, dropout=cfg.dropout))
        self.stage2 = nn.Sequential(*s2)

        # Attention stages on small spatial grids (8x8 then 4x4 for 64x64 inputs).
        heads3 = _pick_heads(w3, preferred=[12, 8, 6, 4, 3, 2, 1])
        heads4 = _pick_heads(w4, preferred=[12, 8, 6, 4, 3, 2, 1])
        s3: list[nn.Module] = [ConvBNAct(w2, w3, kernel_size=3, stride=2, act="silu")]
        s3.extend([Attention2DBlock(w3, num_heads=heads3, dropout=cfg.dropout) for _ in range(d3)])
        self.stage3 = nn.Sequential(*s3)

        s4: list[nn.Module] = [ConvBNAct(w3, w4, kernel_size=3, stride=2, act="silu")]
        s4.extend([Attention2DBlock(w4, num_heads=heads4, dropout=cfg.dropout) for _ in range(d4)])
        self.stage4 = nn.Sequential(*s4)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(w4, int(cfg.num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_coatnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    w = float(width_mult)

    if name in {"coatnet_tiny", "tiny", "coatnet"}:
        widths = (32, 64, 128, 256)
        depths = (2, 2, 2, 2)
    elif name in {"coatnet_small", "small"}:
        widths = (32, 96, 192, 384)
        depths = (2, 3, 3, 3)
    elif name in {"coatnet_base", "base"}:
        widths = (48, 128, 256, 512)
        depths = (3, 4, 4, 4)
    else:
        raise ValueError("Unknown CoAtNet variant. Supported: coatnet_tiny|coatnet_small|coatnet_base")

    widths_scaled = tuple(_c(int(c), w, min_ch=32, divisor=8) for c in widths)
    return CoAtNetClassifier(
        CoAtNetConfig(
            image_size=int(image_size),
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            widths=tuple(map(int, widths_scaled)),
            depths=tuple(map(int, depths)),
            dropout=float(dropout),
        )
    )


# ---------------------------------------------------------------------------
# FNet (Fourier mixing), simplified
# ---------------------------------------------------------------------------


class FNetBlock(nn.Module):
    def __init__(self, dim: int, *, ff_dim: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.ln1 = nn.LayerNorm(d)
        self.drop1 = nn.Dropout(p=float(dropout))
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, int(ff_dim)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(ff_dim), d),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixer: Fourier transform over tokens.
        y = self.ln1(x)
        y = torch.fft.fft(y, dim=1).real
        x = x + self.drop1(y)
        y = self.ff(self.ln2(x))
        x = x + self.drop2(y)
        return x


@dataclass(frozen=True)
class FNetConfig:
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    depth: int
    dropout: float
    num_classes: int


class FNetClassifier(nn.Module):
    def __init__(self, cfg: FNetConfig) -> None:
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
        )
        self.pos = nn.Parameter(torch.zeros(1, int(num_tokens), int(cfg.embed_dim)))
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.blocks = nn.Sequential(
            *[
                FNetBlock(
                    int(cfg.embed_dim),
                    ff_dim=int(cfg.embed_dim) * 4,
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)  # (B, C, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B, T, C)
        x = self.drop(x + self.pos)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def build_fnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name, patch_size = _parse_patch_variant(variant, default_patch_size=8)
    w = float(width_mult)

    if name in {"fnet_tiny", "tiny", "fnet"}:
        embed_dim, depth = 192, 6
    elif name in {"fnet_small", "small"}:
        embed_dim, depth = 256, 8
    elif name in {"fnet_base", "base"}:
        embed_dim, depth = 384, 10
    else:
        raise ValueError("Unknown FNet variant. Supported: fnet_tiny|fnet_small|fnet_base (+ _p*)")

    return FNetClassifier(
        FNetConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=_c(int(embed_dim), w, min_ch=96, divisor=8),
            depth=int(depth),
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    )


__all__ = [
    "build_coatnet_classifier",
    "build_fnet_classifier",
    "build_mobilevit_classifier",
]

