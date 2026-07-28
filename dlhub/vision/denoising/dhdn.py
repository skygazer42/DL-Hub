import torch
import torch.nn.functional as F
from torch import nn


class _HybridDilatedBlock(nn.Module):
    """A lightweight hybrid (normal + dilated) residual block."""

    def __init__(self, channels: int, *, dilation: int = 2) -> None:
        super().__init__()
        c = int(channels)
        d = int(dilation)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if d <= 0:
            raise ValueError("dilation must be > 0")

        self.a1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.a2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

        self.b1 = nn.Conv2d(c, c, kernel_size=3, padding=d, dilation=d, bias=True)
        self.b2 = nn.Conv2d(c, c, kernel_size=3, padding=d, dilation=d, bias=True)

        self.fuse = nn.Conv2d(c * 2, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = F.relu(self.a1(x), inplace=True)
        y1 = self.a2(y1)

        y2 = F.relu(self.b1(x), inplace=True)
        y2 = self.b2(y2)

        y = torch.cat([y1, y2], dim=1)
        y = self.fuse(y)
        return F.relu(x + y, inplace=True)


class DHDN(nn.Module):
    """DHDN-style denoiser (compact-first, pure torch).

    Inspired by "hybrid" denoising networks that mix normal and dilated convolutions to
    capture both local detail and wider context. Output is `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        depth: int = 8,
        dilation: int = 2,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        d = int(depth)
        dil = int(dilation)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")
        if dil <= 0:
            raise ValueError("dilation must be > 0")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.blocks = nn.Sequential(*[_HybridDilatedBlock(f, dilation=dil) for _ in range(d)])
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
    "dhdn_tiny": {"features": 32, "depth": 4, "dilation": 2},
    "dhdn_small": {"features": 48, "depth": 6, "dilation": 2},
    "dhdn_base": {"features": 64, "depth": 10, "dilation": 2},
}


def build_dhdn_denoiser(*, in_channels: int, variant: str = "dhdn_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DHDN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DHDN(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        depth=int(spec["depth"]),
        dilation=int(spec["dilation"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_dhdn_denoiser(in_channels=1, variant="dhdn_tiny")
    y = m(noisy)
    print("dhdn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
