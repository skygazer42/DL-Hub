"""DID-MDN (Density-aware Image Deraining with Multi-stream Dense Network) - compact-first.

Reference (original idea):
- "DID-MDN: Dense-Dense Multi-stream Dense Network for Single Image Rain Removal"

Compact interpretation:
- Multi-dilation feature extractor predicts rain-density attention and rain residual.
- Output is shape-preserving: `x - attention * residual`.
"""

import torch
import torch.nn.functional as F
from torch import nn


class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth: int, *, dilation: int) -> None:
        super().__init__()
        c_in = int(in_channels)
        g = int(growth)
        d = int(dilation)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if g <= 0:
            raise ValueError("growth must be > 0")
        if d <= 0:
            raise ValueError("dilation must be > 0")
        self.conv = nn.Conv2d(c_in, g, kernel_size=3, padding=d, dilation=d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv(x), inplace=True)
        return torch.cat([x, y], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, channels: int, *, layers: int, growth: int) -> None:
        super().__init__()
        c = int(channels)
        n = int(layers)
        g = int(growth)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if n <= 0:
            raise ValueError("layers must be > 0")
        if g <= 0:
            raise ValueError("growth must be > 0")
        mods: list[nn.Module] = []
        cur = c
        dilations = (1, 2, 3)
        for i in range(n):
            mods.append(_DenseLayer(cur, g, dilation=dilations[i % len(dilations)]))
            cur += g
        self.layers = nn.ModuleList(mods)
        self.compress = nn.Conv2d(cur, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        y = self.compress(y)
        return F.relu(x + y, inplace=True)


class DIDMDN(nn.Module):
    """Compact DID-MDN-style derainer with rain-density attention."""

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 24,
        blocks: int = 3,
        dense_layers: int = 3,
        growth: int = 8,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        b = int(blocks)
        n = int(dense_layers)
        g = int(growth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if b <= 0:
            raise ValueError("blocks must be > 0")
        if n <= 0:
            raise ValueError("dense_layers must be > 0")
        if g <= 0:
            raise ValueError("growth must be > 0")

        self.in_channels = c_in
        self.stem = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(
            *[_DenseBlock(f, layers=n, growth=g) for _ in range(b)],
        )
        self.residual_head = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)
        self.attn_head = nn.Conv2d(f, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {int(x.shape[1])} channels"
            )

        feat = F.relu(self.stem(x), inplace=True)
        feat = self.body(feat)
        residual = self.residual_head(feat)
        rain_density = torch.sigmoid(self.attn_head(feat))
        return x - rain_density * residual


_VARIANTS: dict[str, dict] = {
    "did_mdn_tiny": {"features": 16, "blocks": 2, "dense_layers": 2, "growth": 6},
    "did_mdn_small": {"features": 24, "blocks": 3, "dense_layers": 3, "growth": 8},
    "did_mdn_base": {"features": 32, "blocks": 4, "dense_layers": 4, "growth": 10},
}


def build_did_mdn_denoiser(*, in_channels: int, variant: str = "did_mdn_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DID-MDN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DIDMDN(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        blocks=int(spec["blocks"]),
        dense_layers=int(spec["dense_layers"]),
        growth=int(spec["growth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_did_mdn_denoiser(in_channels=3, variant="did_mdn_tiny")
    y = m(x)
    print("did_mdn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
