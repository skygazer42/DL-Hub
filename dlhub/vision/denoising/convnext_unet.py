from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors (per-pixel)."""

    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (depthwise 7x7 + LN + MLP), simplified for toy restoration."""

    def __init__(self, dim: int, *, mlp_ratio: int = 4) -> None:
        super().__init__()
        d = int(dim)
        r = int(mlp_ratio)
        if d <= 0:
            raise ValueError("dim must be > 0")
        if r <= 1:
            raise ValueError("mlp_ratio must be > 1")

        self.dwconv = nn.Conv2d(d, d, kernel_size=7, padding=3, groups=d, bias=True)
        self.norm = LayerNorm2d(d)
        hidden = d * r
        self.pw1 = nn.Conv2d(d, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, d, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, d, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x)
        y = self.norm(y)
        y = self.pw2(self.act(self.pw1(y)))
        return x + y * self.gamma


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, depth: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=2, stride=2, bias=True)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(int(out_ch)) for _ in range(int(depth))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.blocks(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, depth: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2, bias=True)
        self.reduce = nn.Conv2d(int(out_ch) * 2, int(out_ch), kernel_size=1, bias=True)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(int(out_ch)) for _ in range(int(depth))])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.reduce(torch.cat([skip, x], dim=1))
        return self.blocks(x)


class ConvNeXtUNet(nn.Module):
    """ConvNeXt-U-Net style denoiser (toy-first, pure torch).

    Encoder-decoder with ConvNeXt blocks. Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 24,
        depths: tuple[int, int, int, int] = (1, 1, 2, 2),
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        ds = tuple(int(x) for x in depths)
        if len(ds) != 4:
            raise ValueError("depths must be a 4-tuple")
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if base < 8:
            raise ValueError("base_channels must be >= 8")

        dims = (base, base * 2, base * 4, base * 8)

        self.intro = nn.Conv2d(c_in, dims[0], kernel_size=3, padding=1, bias=True)
        self.enc0 = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(ds[0])])
        self.down1 = _Down(dims[0], dims[1], depth=ds[1])
        self.down2 = _Down(dims[1], dims[2], depth=ds[2])
        self.down3 = _Down(dims[2], dims[3], depth=ds[3])

        self.up3 = _Up(dims[3], dims[2], depth=max(1, ds[2] // 2))
        self.up2 = _Up(dims[2], dims[1], depth=max(1, ds[1] // 2))
        self.up1 = _Up(dims[1], dims[0], depth=max(1, ds[0] // 2))

        self.outro = nn.Conv2d(dims[0], c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        inp = x

        x0 = self.enc0(F.relu(self.intro(x), inplace=True))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        y = self.up3(x3, x2)
        y = self.up2(y, x1)
        y = self.up1(y, x0)

        residual = self.outro(y)
        return inp - residual


_VARIANTS: dict[str, dict] = {
    "convnext_unet_tiny": {"base_channels": 16, "depths": (1, 1, 1, 1)},
    "convnext_unet_small": {"base_channels": 24, "depths": (1, 1, 2, 2)},
    "convnext_unet_base": {"base_channels": 32, "depths": (2, 2, 3, 3)},
}


def build_convnext_unet_denoiser(*, in_channels: int, variant: str = "convnext_unet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ConvNeXtUNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ConvNeXtUNet(
        in_channels=int(in_channels),
        base_channels=int(spec["base_channels"]),
        depths=tuple(int(x) for x in spec["depths"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_convnext_unet_denoiser(in_channels=1, variant="convnext_unet_tiny")
    y = m(noisy)
    print("convnext_unet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

