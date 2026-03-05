from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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


class _Fuse(nn.Module):
    """Fuse multi-scale features into a single tensor at a target resolution."""

    def __init__(self, in_channels_list: list[int], out_ch: int) -> None:
        super().__init__()
        oc = int(out_ch)
        self.proj = nn.ModuleList([nn.Conv2d(int(c), oc, kernel_size=1, bias=True) for c in in_channels_list])
        self.conv = _DoubleConv(oc * len(in_channels_list), oc)

    def forward(self, feats: list[torch.Tensor], *, size_hw: tuple[int, int]) -> torch.Tensor:
        if len(feats) != len(self.proj):
            raise ValueError("feature/proj length mismatch")
        h, w = int(size_hw[0]), int(size_hw[1])
        outs: list[torch.Tensor] = []
        for x, p in zip(feats, self.proj, strict=True):
            y = F.relu(p(x), inplace=True)
            if y.shape[-2:] != (h, w):
                y = F.interpolate(y, size=(h, w), mode="nearest")
            outs.append(y)
        return self.conv(torch.cat(outs, dim=1))


class UNet3Plus(nn.Module):
    """U-Net 3+ (UNet3+) style denoiser (toy-first, pure torch).

    UNet3+ uses full-scale skip connections: each decoder stage fuses features from multiple encoder depths.
    This toy-first implementation keeps the spirit (multi-scale fusion) while remaining compact.

    Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if base < 8:
            raise ValueError("base_channels must be >= 8")

        c0, c1, c2, c3 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc0 = _DoubleConv(c_in, c0)
        self.enc1 = _DoubleConv(c0, c1)
        self.enc2 = _DoubleConv(c1, c2)
        self.enc3 = _DoubleConv(c2, c3)

        # Decoder fusions (multi-scale)
        self.fuse2 = _Fuse([c0, c1, c2, c3], out_ch=c2)  # target H/4
        self.fuse1 = _Fuse([c0, c1, c2, c3], out_ch=c1)  # target H/2
        self.fuse0 = _Fuse([c0, c1, c2, c3], out_ch=c0)  # target H

        self.outc = nn.Conv2d(c0, c_in, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        inp = x

        x0 = self.enc0(x)  # H
        x1 = self.enc1(self.pool(x0))  # H/2
        x2 = self.enc2(self.pool(x1))  # H/4
        x3 = self.enc3(self.pool(x2))  # H/8

        # Multi-scale decode with repeated fusion
        d2 = self.fuse2([x0, x1, x2, x3], size_hw=x2.shape[-2:])  # H/4
        d1 = self.fuse1([x0, x1, d2, x3], size_hw=x1.shape[-2:])  # H/2 (replace x2 with d2)
        d0 = self.fuse0([x0, d1, d2, x3], size_hw=x0.shape[-2:])  # H

        residual = self.outc(d0)
        return inp - residual


_VARIANTS: dict[str, dict] = {
    "unet3plus_tiny": {"base_channels": 12},
    "unet3plus_small": {"base_channels": 16},
    "unet3plus_base": {"base_channels": 24},
}


def build_unet3plus_denoiser(*, in_channels: int, variant: str = "unet3plus_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown UNet3Plus variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return UNet3Plus(in_channels=int(in_channels), base_channels=int(spec["base_channels"]))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_unet3plus_denoiser(in_channels=1, variant="unet3plus_tiny")
    y = m(noisy)
    print("unet3plus_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

