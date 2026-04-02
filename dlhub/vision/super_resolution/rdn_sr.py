from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import PixelShuffleUpsampler, check_low_res_image, validate_upscale_factor


class _ResidualDenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(int(in_channels), int(growth), kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.conv(x))
        return torch.cat([x, y], dim=1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, *, growth: int, num_layers: int) -> None:
        super().__init__()
        c = int(channels)
        g = int(growth)
        depth = int(num_layers)
        if c <= 0 or g <= 0 or depth <= 0:
            raise ValueError("channels, growth, and num_layers must be > 0")

        layers: list[nn.Module] = []
        fused_channels = c
        for _ in range(depth):
            layers.append(_ResidualDenseLayer(fused_channels, g))
            fused_channels += g
        self.layers = nn.ModuleList(layers)
        self.fuse = nn.Conv2d(fused_channels, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + self.fuse(y)


class RDNSR(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        features: int,
        num_blocks: int,
        num_layers: int,
        growth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        blocks = int(num_blocks)
        layers = int(num_layers)
        g = int(growth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if blocks <= 0 or layers <= 0 or g <= 0:
            raise ValueError("num_blocks, num_layers, and growth must be > 0")

        self.sfe1 = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.sfe2 = nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True)
        self.rdbs = nn.ModuleList(
            [ResidualDenseBlock(f, growth=g, num_layers=layers) for _ in range(blocks)]
        )
        self.gff1 = nn.Conv2d(f * blocks, f, kernel_size=1, bias=True)
        self.gff2 = nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True)
        self.upsample = PixelShuffleUpsampler(f, upscale_factor=2)
        self.tail = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, low_res: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_low_res_image(low_res)
        f1 = F.gelu(self.sfe1(x))
        y = F.gelu(self.sfe2(f1))

        feats: list[torch.Tensor] = []
        for block in self.rdbs:
            y = block(y)
            feats.append(y)

        y = F.gelu(self.gff1(torch.cat(feats, dim=1)))
        y = self.gff2(y) + f1
        y = F.gelu(self.upsample(y))
        sr = self.tail(y)
        return {"sr": sr}


_VARIANTS: dict[str, dict[str, int]] = {
    "rdn_sr_tiny": {"features": 24, "num_blocks": 3, "num_layers": 3, "growth": 12},
    "rdn_sr_small": {"features": 32, "num_blocks": 4, "num_layers": 4, "growth": 16},
    "rdn_sr_base": {"features": 48, "num_blocks": 6, "num_layers": 5, "growth": 24},
}


def build_rdn_sr_super_resolver(
    *,
    in_channels: int,
    variant: str = "rdn_sr_small",
    upscale_factor: int = 2,
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del dropout
    validate_upscale_factor(upscale_factor)

    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RDN-SR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    features = max(8, int(int(spec["features"]) * float(width_mult)))
    growth = max(4, int(int(spec["growth"]) * float(width_mult)))
    return RDNSR(
        in_channels=int(in_channels),
        features=features,
        num_blocks=int(spec["num_blocks"]),
        num_layers=int(spec["num_layers"]),
        growth=growth,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_rdn_sr_super_resolver(in_channels=3, variant="rdn_sr_tiny")
    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    print("rdn_sr_tiny", tuple(y["sr"].shape))
