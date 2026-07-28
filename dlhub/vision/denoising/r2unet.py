import torch
import torch.nn.functional as F
from torch import nn


class _RecurrentConv(nn.Module):
    def __init__(self, channels: int, *, t: int = 2) -> None:
        super().__init__()
        c = int(channels)
        tt = int(t)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if tt <= 0:
            raise ValueError("t must be > 0")
        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.t = tt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(self.t):
            h = F.relu(self.conv(x + h), inplace=True)
        return h


class _RRCNNBlock(nn.Module):
    """Recurrent Residual CNN block used in R2U-Net (simplified)."""

    def __init__(self, in_ch: int, out_ch: int, *, t: int = 2) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        if c_in <= 0 or c_out <= 0:
            raise ValueError("channels must be > 0")
        self.in_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)
        self.rc1 = _RecurrentConv(c_out, t=int(t))
        self.rc2 = _RecurrentConv(c_out, t=int(t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.in_conv(x), inplace=True)
        y = self.rc1(y)
        y = self.rc2(y)
        return F.relu(y + F.relu(self.in_conv(x), inplace=False), inplace=True)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, t: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = _RRCNNBlock(int(in_ch), int(out_ch), t=int(t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, t: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2, bias=True)
        self.block = _RRCNNBlock(int(out_ch) * 2, int(out_ch), t=int(t))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class R2UNet(nn.Module):
    """R2U-Net (Recurrent Residual U-Net) denoiser (compact-first, pure torch).

    Uses RRCNN blocks in a U-Net encoder-decoder. Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 24,
        levels: int = 4,
        t: int = 2,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        lv = int(levels)
        tt = int(t)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if base < 8:
            raise ValueError("base_channels must be >= 8")
        if lv < 2:
            raise ValueError("levels must be >= 2")
        if tt <= 0:
            raise ValueError("t must be > 0")

        self.inc = _RRCNNBlock(c_in, base, t=tt)

        downs: list[nn.Module] = []
        ch = base
        for _ in range(lv - 1):
            downs.append(_Down(ch, ch * 2, t=tt))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        ups: list[nn.Module] = []
        for _ in range(lv - 1):
            ups.append(_Up(ch, ch // 2, t=tt))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.outc = nn.Conv2d(base, c_in, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        inp = x

        skips: list[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        residual = self.outc(x)
        return inp - residual


_VARIANTS: dict[str, dict] = {
    "r2unet_tiny": {"base_channels": 16, "levels": 3, "t": 2},
    "r2unet_small": {"base_channels": 24, "levels": 4, "t": 2},
    "r2unet_base": {"base_channels": 32, "levels": 4, "t": 3},
}


def build_r2unet_denoiser(*, in_channels: int, variant: str = "r2unet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown R2UNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return R2UNet(
        in_channels=int(in_channels),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
        t=int(spec["t"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_r2unet_denoiser(in_channels=1, variant="r2unet_tiny")
    y = m(noisy)
    print("r2unet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
