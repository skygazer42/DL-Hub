from dataclasses import dataclass

import torch
from torch import nn

from .transformers import TransformerEncoderBlock


def _make_divisible(v: int, divisor: int = 8) -> int:
    d = int(divisor)
    if d <= 0:
        raise ValueError("divisor must be > 0")
    x = int(v)
    if x <= 0:
        return d
    return int((x + d - 1) // d * d)


def _c(ch: int, width_mult: float, *, min_ch: int = 32, divisor: int = 8) -> int:
    v = max(int(min_ch), int(round(int(ch) * float(width_mult))))
    return _make_divisible(v, int(divisor))


def _pick_heads(embed_dim: int, preferred: list[int]) -> int:
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


class PVTStage(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        h, w = int(input_resolution[0]), int(input_resolution[1])
        if h <= 0 or w <= 0:
            raise ValueError("input_resolution must be positive")
        if int(depth) <= 0:
            raise ValueError("depth must be > 0")

        self.input_resolution = (h, w)
        self.proj = nn.Conv2d(
            int(in_ch),
            int(embed_dim),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=0,
            bias=True,
        )
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(embed_dim)))
        self.drop = nn.Dropout(p=float(dropout))
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=int(embed_dim),
                    num_heads=int(num_heads),
                    ff_dim=int(embed_dim) * 4,
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.proj(x)
        b, c, h, w = x.shape
        if (h, w) != self.input_resolution:
            raise ValueError(f"Expected resolution {self.input_resolution}, got {(h, w)}")

        t = x.flatten(2).transpose(1, 2)  # (B, T, C)
        t = self.drop(t + self.pos)
        t = self.blocks(t)
        x = t.transpose(1, 2).contiguous().view(b, c, h, w)
        return x


@dataclass(frozen=True)
class PVTConfig:
    image_size: int
    patch_size: int
    in_channels: int
    embed_dims: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    num_heads: tuple[int, int, int, int]
    dropout: float
    num_classes: int


class PVTClassifier(nn.Module):
    def __init__(self, cfg: PVTConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")

        grid = int(cfg.image_size) // int(cfg.patch_size)
        # We downsample by 2 at each subsequent stage (4 stages total).
        if grid % 8 != 0:
            raise ValueError(
                "PVT requires (image_size / patch_size) divisible by 8 for 4-stage pyramid"
            )

        self.cfg = cfg

        g1 = grid
        g2 = g1 // 2
        g3 = g2 // 2
        g4 = g3 // 2

        dims = tuple(map(int, cfg.embed_dims))
        depths = tuple(map(int, cfg.depths))
        heads = tuple(map(int, cfg.num_heads))

        self.stage1 = PVTStage(
            in_ch=int(cfg.in_channels),
            embed_dim=dims[0],
            kernel_size=int(cfg.patch_size),
            stride=int(cfg.patch_size),
            input_resolution=(g1, g1),
            depth=depths[0],
            num_heads=heads[0],
            dropout=float(cfg.dropout),
        )
        self.stage2 = PVTStage(
            in_ch=dims[0],
            embed_dim=dims[1],
            kernel_size=2,
            stride=2,
            input_resolution=(g2, g2),
            depth=depths[1],
            num_heads=heads[1],
            dropout=float(cfg.dropout),
        )
        self.stage3 = PVTStage(
            in_ch=dims[1],
            embed_dim=dims[2],
            kernel_size=2,
            stride=2,
            input_resolution=(g3, g3),
            depth=depths[2],
            num_heads=heads[2],
            dropout=float(cfg.dropout),
        )
        self.stage4 = PVTStage(
            in_ch=dims[2],
            embed_dim=dims[3],
            kernel_size=2,
            stride=2,
            input_resolution=(g4, g4),
            depth=depths[3],
            num_heads=heads[3],
            dropout=float(cfg.dropout),
        )

        self.norm = nn.LayerNorm(dims[-1])
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Linear(dims[-1], int(cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        if int(x.shape[2]) != int(self.cfg.image_size) or int(x.shape[3]) != int(
            self.cfg.image_size
        ):
            raise ValueError(
                f"Expected image_size={self.cfg.image_size}, got HxW={tuple(x.shape[2:])}"
            )

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        x = self.drop(x)
        return self.head(x)


def build_pvt_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name, patch_size = _parse_patch_variant(variant, default_patch_size=4)
    w = float(width_mult)

    if name in {"pvt_tiny", "tiny", "pvt"}:
        embed_dims = (64, 128, 256, 512)
        depths = (2, 2, 2, 2)
    elif name in {"pvt_small", "small"}:
        embed_dims = (64, 128, 320, 512)
        depths = (2, 2, 3, 2)
    elif name in {"pvt_base", "base"}:
        embed_dims = (96, 192, 384, 768)
        depths = (2, 2, 4, 2)
    else:
        raise ValueError("Unknown PVT variant. Supported: pvt_tiny|pvt_small|pvt_base (+ _p*)")

    embed_dims_scaled = tuple(_c(int(d), w, min_ch=64, divisor=8) for d in embed_dims)
    heads = tuple(_pick_heads(d, preferred=[12, 8, 6, 4, 3, 2, 1]) for d in embed_dims_scaled)

    return PVTClassifier(
        PVTConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dims=tuple(map(int, embed_dims_scaled)),
            depths=tuple(map(int, depths)),
            num_heads=tuple(map(int, heads)),
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    )


__all__ = ["build_pvt_classifier"]
