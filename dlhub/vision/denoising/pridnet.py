from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _SEBlock(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        c = int(channels)
        r = int(reduction)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if r <= 0:
            raise ValueError("reduction must be > 0")
        hidden = max(4, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class _PyramidBlock(nn.Module):
    """A lightweight PRIDNet-inspired pyramid feature + attention block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)

        self.b1 = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.b2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.b3a = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.b3b = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

        self.fuse = nn.Conv2d(c * 3, c, kernel_size=1, bias=True)
        self.se = _SEBlock(c, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = F.relu(self.b1(x), inplace=True)
        b2 = F.relu(self.b2(x), inplace=True)
        b3 = F.relu(self.b3a(x), inplace=True)
        b3 = F.relu(self.b3b(b3), inplace=True)

        y = torch.cat([b1, b2, b3], dim=1)
        y = F.relu(self.fuse(y), inplace=True)
        y = self.se(y)
        return F.relu(x + y, inplace=True)


class PRIDNet(nn.Module):
    """PRIDNet-style progressive residual denoiser (toy-first, pure torch).

    This is a simplified PRIDNet-inspired model:
    - Pyramid feature extraction per block (multi-receptive-field branches)
    - Channel attention (SE)
    - Residual prediction: output is `x - residual`
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        b = int(num_blocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if b <= 0:
            raise ValueError("num_blocks must be > 0")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.blocks = nn.Sequential(*[_PyramidBlock(f) for _ in range(b)])
        self.out_conv = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = F.relu(self.in_conv(x), inplace=True)
        y = self.blocks(y)
        residual = self.out_conv(y)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "pridnet_tiny": {"features": 32, "num_blocks": 4},
    "pridnet_small": {"features": 48, "num_blocks": 6},
    "pridnet_base": {"features": 64, "num_blocks": 8},
}


def build_pridnet_denoiser(*, in_channels: int, variant: str = "pridnet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PRIDNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return PRIDNet(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        num_blocks=int(spec["num_blocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_pridnet_denoiser(in_channels=3, variant="pridnet_tiny")
    y = m(noisy)
    print("pridnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

