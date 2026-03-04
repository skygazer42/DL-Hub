from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels


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


class SpatialGatingUnit(nn.Module):
    def __init__(self, *, num_tokens: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.proj = nn.Linear(int(num_tokens), int(num_tokens))
        self.drop = nn.Dropout(p=float(dropout))

        # Near-identity init: proj(v) ~= 1 => u * 1.
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


class GMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(image_size) % int(patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if int(depth) <= 0:
            raise ValueError("depth must be > 0")

        grid = int(image_size) // int(patch_size)
        num_tokens = grid * grid

        self.patch_embed = nn.Conv2d(
            int(in_channels),
            int(embed_dim),
            kernel_size=int(patch_size),
            stride=int(patch_size),
            padding=0,
        )
        self.blocks = nn.Sequential(
            *[
                GMLPBlock(int(embed_dim), num_tokens=int(num_tokens), ff_mult=int(ff_mult), dropout=float(dropout))
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(embed_dim))
        self.head = nn.Linear(int(embed_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, T, C)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


_SPECS: dict[str, tuple[int, int]] = {
    "gmlp_tiny": (128, 6),
    "gmlp_small": (256, 8),
    "gmlp_base": (384, 10),
    "tiny": (128, 6),
    "small": (256, 8),
    "base": (384, 10),
    "gmlp": (128, 6),
}


def build_gmlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "gmlp_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name, patch_size = _parse_patch_variant(variant, default_patch_size=8)
    if name not in _SPECS:
        raise ValueError("Unknown gMLP variant. Supported: gmlp_tiny|gmlp_small|gmlp_base (+ _p*)")
    base_dim, depth = _SPECS[name]
    embed_dim = scale_channels(int(base_dim), float(width_mult), min_ch=64, divisor=8)
    return GMLPClassifier(
        image_size=int(image_size),
        patch_size=int(patch_size),
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        embed_dim=int(embed_dim),
        depth=int(depth),
        ff_mult=4,
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["gmlp_tiny", "gmlp_small", "gmlp_base", "gmlp_tiny_p16"]:
        m = build_gmlp_classifier(in_channels=3, num_classes=10, image_size=64, variant=v)
        y = m(x)
        print(v, tuple(y.shape))

