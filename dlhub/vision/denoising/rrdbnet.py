
import torch
from torch import nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(int(in_ch), int(growth), kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.leaky_relu(self.conv(x), negative_slope=0.2, inplace=True)
        return torch.cat([x, y], dim=1)


class _ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) used inside RRDB."""

    def __init__(self, channels: int, *, growth: int = 32, num_layers: int = 5, res_scale: float = 0.2) -> None:
        super().__init__()
        c = int(channels)
        g = int(growth)
        l = int(num_layers)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if g <= 0:
            raise ValueError("growth must be > 0")
        if l <= 0:
            raise ValueError("num_layers must be > 0")
        self.res_scale = float(res_scale)

        layers: list[nn.Module] = []
        ch = c
        for _ in range(l):
            layers.append(_DenseLayer(ch, g))
            ch += g
        self.layers = nn.ModuleList(layers)
        self.lff = nn.Conv2d(ch, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        y = self.lff(y)
        return x + y * self.res_scale


class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB) as used in ESRGAN/RRDBNet."""

    def __init__(self, channels: int, *, growth: int = 32, res_scale: float = 0.2) -> None:
        super().__init__()
        c = int(channels)
        self.rdb1 = _ResidualDenseBlock(c, growth=int(growth), num_layers=5, res_scale=float(res_scale))
        self.rdb2 = _ResidualDenseBlock(c, growth=int(growth), num_layers=5, res_scale=float(res_scale))
        self.rdb3 = _ResidualDenseBlock(c, growth=int(growth), num_layers=5, res_scale=float(res_scale))
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rdb1(x)
        y = self.rdb2(y)
        y = self.rdb3(y)
        return x + y * self.res_scale


class RRDBNet(nn.Module):
    """RRDBNet-style denoiser (toy-first, pure torch).

    RRDBNet is a strong CNN backbone originally popularized for SR (ESRGAN).
    Here we keep resolution and predict a residual/noise map, returning `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        num_blocks: int = 8,
        growth: int = 32,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        b = int(num_blocks)
        g = int(growth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if b <= 0:
            raise ValueError("num_blocks must be > 0")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.trunk = nn.Sequential(*[RRDB(f, growth=g, res_scale=float(res_scale)) for _ in range(b)])
        self.trunk_conv = nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True)
        self.out_conv = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        feat = F.leaky_relu(self.in_conv(x), negative_slope=0.2, inplace=True)
        y = self.trunk(feat)
        y = self.trunk_conv(y)
        y = y + feat
        residual = self.out_conv(F.leaky_relu(y, negative_slope=0.2, inplace=True))
        return x - residual


_VARIANTS: dict[str, dict] = {
    "rrdbnet_tiny": {"features": 32, "num_blocks": 3, "growth": 16, "res_scale": 0.2},
    "rrdbnet_small": {"features": 48, "num_blocks": 6, "growth": 24, "res_scale": 0.2},
    "rrdbnet_base": {"features": 64, "num_blocks": 10, "growth": 32, "res_scale": 0.2},
}


def build_rrdbnet_denoiser(*, in_channels: int, variant: str = "rrdbnet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RRDBNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RRDBNet(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        num_blocks=int(spec["num_blocks"]),
        growth=int(spec["growth"]),
        res_scale=float(spec["res_scale"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_rrdbnet_denoiser(in_channels=3, variant="rrdbnet_tiny")
    y = m(noisy)
    print("rrdbnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

