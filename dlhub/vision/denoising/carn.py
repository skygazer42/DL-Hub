from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(x + y, inplace=True)


class _CascadingBlock(nn.Module):
    """CARN-style cascading block: multiple residual blocks + 1x1 fusion."""

    def __init__(self, channels: int, *, num_resblocks: int) -> None:
        super().__init__()
        c = int(channels)
        n = int(num_resblocks)
        if n <= 0:
            raise ValueError("num_resblocks must be > 0")

        self.blocks = nn.ModuleList([_ResBlock(c) for _ in range(n)])
        self.fuse = nn.Conv2d(c * n, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: list[torch.Tensor] = []
        y = x
        for blk in self.blocks:
            y = blk(y)
            feats.append(y)
        fused = self.fuse(torch.cat(feats, dim=1))
        return F.relu(x + fused, inplace=True)


class CARN(nn.Module):
    """CARN (Cascading Residual Network) adapted for denoising (toy-first, pure torch).

    Uses multiple cascading blocks and global feature fusion via concatenation + 1x1 conv.
    Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        num_cascades: int = 4,
        num_resblocks: int = 3,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        c = int(num_cascades)
        r = int(num_resblocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if c <= 0:
            raise ValueError("num_cascades must be > 0")
        if r <= 0:
            raise ValueError("num_resblocks must be > 0")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)

        self.cascades = nn.ModuleList([_CascadingBlock(f, num_resblocks=r) for _ in range(c)])
        self.gff = nn.Conv2d(f * c, f, kernel_size=1, bias=True)

        self.out_conv = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        feat = F.relu(self.in_conv(x), inplace=True)
        outs: list[torch.Tensor] = []
        y = feat
        for blk in self.cascades:
            y = blk(y)
            outs.append(y)
        y = self.gff(torch.cat(outs, dim=1))
        y = y + feat
        residual = self.out_conv(F.relu(y, inplace=True))
        return x - residual


_VARIANTS: dict[str, dict] = {
    "carn_tiny": {"features": 32, "num_cascades": 2, "num_resblocks": 2},
    "carn_small": {"features": 48, "num_cascades": 3, "num_resblocks": 3},
    "carn_base": {"features": 64, "num_cascades": 4, "num_resblocks": 3},
}


def build_carn_denoiser(*, in_channels: int, variant: str = "carn_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CARN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return CARN(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        num_cascades=int(spec["num_cascades"]),
        num_resblocks=int(spec["num_resblocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_carn_denoiser(in_channels=1, variant="carn_tiny")
    y = m(noisy)
    print("carn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

