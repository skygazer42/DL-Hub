from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from ._utils import pad_to_multiple, unpad


def _group_norm(ch: int) -> nn.GroupNorm:
    c = int(ch)
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for a scalar tensor (B,) -> (B, dim)."""

    if x.ndim != 1:
        raise ValueError(f"Expected shape (B,), got {tuple(x.shape)}")
    d = int(dim)
    if d <= 0:
        raise ValueError("dim must be > 0")
    if d % 2 != 0:
        raise ValueError("dim must be even")

    half = d // 2
    device = x.device
    # log-spaced frequencies
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device, dtype=x.dtype))
    args = x[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, time_dim: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        tdim = int(time_dim)
        if c_in <= 0 or c_out <= 0:
            raise ValueError("channels must be > 0")
        if tdim <= 0:
            raise ValueError("time_dim must be > 0")

        self.norm1 = _group_norm(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)

        self.time_proj = nn.Linear(tdim, c_out, bias=True)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (B, time_dim)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DDPMUNet(nn.Module):
    """A small noise-conditioned U-Net used by diffusion models (toy-first).

    This module predicts the *noise / residual*. A wrapper converts it into a denoiser.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 48,
        levels: int = 3,
        blocks_per_level: int = 2,
        time_dim: int | None = None,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        lv = int(levels)
        bpl = int(blocks_per_level)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if base < 8:
            raise ValueError("base_channels must be >= 8")
        if lv < 2:
            raise ValueError("levels must be >= 2")
        if bpl <= 0:
            raise ValueError("blocks_per_level must be > 0")

        # Time embedding
        tdim = int(time_dim) if time_dim is not None else base * 4
        if tdim % 2 != 0:
            tdim += 1
        self.time_dim = int(tdim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Input embedding
        self.in_conv = nn.Conv2d(c_in, base, kernel_size=3, padding=1, bias=True)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_channels: list[int] = []
        ch = base
        for level in range(lv - 1):
            blocks: list[nn.Module] = []
            for _ in range(bpl):
                blocks.append(ResBlock(ch, ch, time_dim=self.time_dim))
            self.enc_blocks.append(nn.ModuleList(blocks))
            enc_channels.append(ch)
            self.downs.append(Downsample(ch))
            ch *= 2
            # widen after downsample
            self.downs.append(nn.Conv2d(ch // 2, ch, kernel_size=1, bias=True))

        # Bottleneck
        bott: list[nn.Module] = []
        for _ in range(bpl):
            bott.append(ResBlock(ch, ch, time_dim=self.time_dim))
        self.bottleneck = nn.ModuleList(bott)

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for skip_ch in reversed(enc_channels):
            self.ups.append(Upsample(ch))
            # reduce channels after concat
            self.ups.append(nn.Conv2d(ch + skip_ch, skip_ch, kernel_size=1, bias=True))
            ch = skip_ch
            blocks: list[nn.Module] = []
            for _ in range(max(1, bpl)):
                blocks.append(ResBlock(ch, ch, time_dim=self.time_dim))
            self.dec_blocks.append(nn.ModuleList(blocks))

        self.out_norm = _group_norm(base)
        self.out_conv = nn.Conv2d(base, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor, *, sigma: float) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        # Scalar sigma -> embedding
        sig = float(sigma)
        t = torch.full((x.shape[0],), sig, device=x.device, dtype=x.dtype)
        t = self.time_mlp(sinusoidal_embedding(t, self.time_dim))

        h = self.in_conv(x)

        skips: list[torch.Tensor] = []
        down_iter = iter(self.downs)
        for blocks in self.enc_blocks:
            for blk in blocks:
                h = blk(h, t)
            skips.append(h)
            # downsample: conv stride2 then widen
            h = next(down_iter)(h)
            h = next(down_iter)(h)

        for blk in self.bottleneck:
            h = blk(h, t)

        for up, reduce, blocks, skip in zip(self.ups[0::2], self.ups[1::2], self.dec_blocks, reversed(skips), strict=True):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = reduce(torch.cat([h, skip], dim=1))
            for blk in blocks:
                h = blk(h, t)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


class DDPMUNetDenoiser(nn.Module):
    """A diffusion-style denoiser wrapper: noisy -> denoised (predict residual and subtract)."""

    def __init__(
        self,
        backbone: DDPMUNet,
        *,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.sigma = float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        # pad to 2^(levels-1); lv is stored implicitly via network depth. We use 8 as safe default.
        x_pad, pad_hw = pad_to_multiple(x, 8, mode="reflect")
        res = self.backbone(x_pad, sigma=float(self.sigma))
        y = x_pad - res
        return unpad(y, pad_hw)


_VARIANTS: dict[str, dict] = {
    "ddpm_unet_tiny": {"base": 24, "levels": 3, "blocks": 1},
    "ddpm_unet_small": {"base": 32, "levels": 3, "blocks": 2},
    "ddpm_unet_base": {"base": 48, "levels": 4, "blocks": 2},
}


def build_ddpm_unet_denoiser(
    *,
    in_channels: int,
    sigma: float = 0.1,
    variant: str = "ddpm_unet_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DDPM-UNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    backbone = DDPMUNet(
        in_channels=int(in_channels),
        base_channels=int(spec["base"]),
        levels=int(spec["levels"]),
        blocks_per_level=int(spec["blocks"]),
    )
    return DDPMUNetDenoiser(backbone, sigma=float(sigma))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_ddpm_unet_denoiser(in_channels=1, sigma=0.1, variant="ddpm_unet_tiny")
    y = m(noisy)
    print("ddpm_unet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

