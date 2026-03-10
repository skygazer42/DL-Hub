import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2)
        self.conv = DoubleConv(int(out_ch) * 2, int(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Noise2NoiseUNet(nn.Module):
    """A small U-Net suitable for Noise2Noise training.

    Noise2Noise is primarily a *training* method. This file provides a standard
    denoising U-Net that works well with either:
    - supervised denoising (noisy -> clean), or
    - noise2noise (noisy_1 -> noisy_2).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 32,
        levels: int = 4,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        lv = int(levels)
        if lv < 2:
            raise ValueError(f"levels must be >= 2, got: {levels}")

        self.inc = DoubleConv(c_in, base)

        downs: list[nn.Module] = []
        ch = base
        for _ in range(lv - 1):
            downs.append(Down(ch, ch * 2))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        ups: list[nn.Module] = []
        for _ in range(lv - 1):
            ups.append(Up(ch, ch // 2))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.outc = nn.Conv2d(base, c_in, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        skips: list[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        # We stored skips at every level; during up, skip in reverse excluding the bottleneck.
        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        return self.outc(x)


_VARIANTS: dict[str, dict] = {
    "n2n_unet_tiny": {"base_channels": 16, "levels": 3},
    "n2n_unet_small": {"base_channels": 24, "levels": 4},
    "n2n_unet_base": {"base_channels": 32, "levels": 4},
}


def build_noise2noise_denoiser(
    *,
    in_channels: int,
    variant: str = "n2n_unet_base",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Noise2Noise variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return Noise2NoiseUNet(
        in_channels=int(in_channels),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 1, 64, 64)
    m = build_noise2noise_denoiser(in_channels=1, variant="n2n_unet_tiny")
    y = m(x)
    print("n2n_unet_tiny", tuple(y.shape))
    loss = (y - x).abs().mean()
    loss.backward()
    print("ok")
