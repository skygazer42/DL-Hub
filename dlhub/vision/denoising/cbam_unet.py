from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        c = int(channels)
        r = int(reduction)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if r <= 0:
            raise ValueError("reduction must be > 0")
        hidden = max(4, c // r)

        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, *, kernel_size: int = 7) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 3 or k % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)
        self.attn = CBAM(int(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.pool(x))
        return self.attn(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2, bias=True)
        self.fuse_attn = CBAM(int(out_ch) * 2)
        self.conv = _DoubleConv(int(out_ch) * 2, int(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        x = self.fuse_attn(x)
        return self.conv(x)


class CBAMUNet(nn.Module):
    """CBAM U-Net denoiser (toy-first, pure torch).

    Uses CBAM (channel + spatial attention) inside a U-Net. Predicts a residual/noise map
    and returns `x - residual`.
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
        if lv < 2:
            raise ValueError("levels must be >= 2")
        if base < 8:
            raise ValueError("base_channels must be >= 8")

        self.inc = _DoubleConv(c_in, base)
        self.inc_attn = CBAM(base)

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

        self.outc = nn.Conv2d(base, c_in, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        inp = x

        skips: list[torch.Tensor] = []
        x = self.inc_attn(self.inc(x))
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        residual = self.outc(x)
        return inp - residual


_VARIANTS: dict[str, dict] = {
    "cbam_unet_tiny": {"base_channels": 16, "levels": 3},
    "cbam_unet_small": {"base_channels": 24, "levels": 4},
    "cbam_unet_base": {"base_channels": 32, "levels": 4},
}


def build_cbam_unet_denoiser(*, in_channels: int, variant: str = "cbam_unet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CBAMUNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return CBAMUNet(
        in_channels=int(in_channels),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_cbam_unet_denoiser(in_channels=1, variant="cbam_unet_tiny")
    y = m(noisy)
    print("cbam_unet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

