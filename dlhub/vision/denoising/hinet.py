from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _HINResBlock(nn.Module):
    """Half Instance Normalization residual block (HIN-style, simplified)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        if c_in <= 0 or c_out <= 0:
            raise ValueError("channels must be > 0")
        if c_out % 2 != 0:
            raise ValueError("out_ch must be even for Half Instance Normalization")

        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=True)
        self.inorm = nn.InstanceNorm2d(c_out // 2, affine=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=True)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y1, y2 = y.chunk(2, dim=1)
        y1 = self.inorm(y1)
        y = torch.cat([y1, y2], dim=1)
        y = F.relu(y, inplace=True)
        y = self.conv2(y)
        return F.relu(self.skip(x) + y, inplace=True)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, stride=2, padding=1, bias=True)
        self.block = _HINResBlock(int(out_ch), int(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.down(x), inplace=True)
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2, bias=True)
        self.block = _HINResBlock(int(out_ch) * 2, int(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class HINet(nn.Module):
    """HINet-style denoiser (toy-first, pure torch).

    Simplified encoder-decoder using Half Instance Normalization residual blocks.
    Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 24,
        levels: int = 4,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        lv = int(levels)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if base < 8:
            raise ValueError("base_channels must be >= 8")
        if base % 2 != 0:
            raise ValueError("base_channels must be even for HINet")
        if lv < 2:
            raise ValueError("levels must be >= 2")

        self.in_block = _HINResBlock(c_in, base)

        downs: list[nn.Module] = []
        ch = base
        for _ in range(lv - 1):
            downs.append(_Down(ch, ch * 2))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        ups: list[nn.Module] = []
        for _ in range(lv - 1):
            ups.append(_Up(ch, ch // 2))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.out_conv = nn.Conv2d(base, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        inp = x
        x = self.in_block(x)
        skips: list[torch.Tensor] = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        residual = self.out_conv(x)
        return inp - residual


_VARIANTS: dict[str, dict] = {
    "hinet_tiny": {"base_channels": 16, "levels": 3},
    "hinet_small": {"base_channels": 24, "levels": 4},
    "hinet_base": {"base_channels": 32, "levels": 4},
}


def build_hinet_denoiser(*, in_channels: int, variant: str = "hinet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown HINet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return HINet(
        in_channels=int(in_channels),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_hinet_denoiser(in_channels=1, variant="hinet_tiny")
    y = m(noisy)
    print("hinet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

