import torch
import torch.nn.functional as F
from torch import nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2)
        self.conv = _DoubleConv(int(out_ch) * 2, int(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP), compact-first."""

    def __init__(
        self, in_ch: int, out_ch: int, *, dilations: tuple[int, ...] = (1, 2, 4, 6)
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        dils = tuple(int(d) for d in dilations)
        if any(d <= 0 for d in dils):
            raise ValueError("dilations must be positive ints")

        branches: list[nn.Module] = []
        for d in dils:
            if d == 1:
                branches.append(nn.Conv2d(c_in, c_out, kernel_size=1, bias=True))
            else:
                branches.append(
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=d, dilation=d, bias=True)
                )
        self.branches = nn.ModuleList(branches)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)

        self.fuse = nn.Conv2d(c_out * (len(dils) + 1), c_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feats = [F.relu(branch(x), inplace=True) for branch in self.branches]

        gp = self.global_proj(self.global_pool(x))
        gp = F.interpolate(gp, size=(h, w), mode="nearest")
        feats.append(F.relu(gp, inplace=True))

        y = torch.cat(feats, dim=1)
        return F.relu(self.fuse(y), inplace=True)


class ASPPUNet(nn.Module):
    """U-Net + ASPP bottleneck denoiser (compact-first, pure torch).

    Uses ASPP at the bottleneck to aggregate multi-dilation context.
    Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 24,
        levels: int = 4,
        aspp_dilations: tuple[int, ...] = (1, 2, 4, 6),
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

        downs: list[nn.Module] = []
        ch = base
        for _ in range(lv - 1):
            downs.append(_Down(ch, ch * 2))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        self.aspp = ASPP(ch, ch, dilations=tuple(aspp_dilations))

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
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.aspp(x)

        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        residual = self.outc(x)
        return inp - residual


_VARIANTS: dict[str, dict] = {
    "aspp_unet_tiny": {"base_channels": 16, "levels": 3, "dilations": (1, 2, 4)},
    "aspp_unet_small": {"base_channels": 24, "levels": 4, "dilations": (1, 2, 4, 6)},
    "aspp_unet_base": {"base_channels": 32, "levels": 4, "dilations": (1, 2, 4, 6)},
}


def build_aspp_unet_denoiser(*, in_channels: int, variant: str = "aspp_unet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ASPPUNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ASPPUNet(
        in_channels=int(in_channels),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
        aspp_dilations=tuple(int(d) for d in spec["dilations"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_aspp_unet_denoiser(in_channels=1, variant="aspp_unet_tiny")
    y = m(noisy)
    print("aspp_unet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
