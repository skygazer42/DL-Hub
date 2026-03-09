
import torch
from torch import nn
import torch.nn.functional as F


class _ResidualDenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(int(in_ch), int(growth), kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv(x))
        return torch.cat([x, y], dim=1)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) used in RDN.

    - Dense connectivity inside the block (concat features).
    - Local Feature Fusion (1x1 conv).
    - Local residual connection.
    """

    def __init__(self, channels: int, *, growth: int = 32, num_layers: int = 6) -> None:
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

        layers: list[nn.Module] = []
        ch = c
        for _ in range(l):
            layers.append(_ResidualDenseLayer(ch, g))
            ch += g
        self.layers = nn.ModuleList(layers)
        self.lff = nn.Conv2d(ch, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        y = self.lff(y)
        return x + y


class RDN(nn.Module):
    """RDN (Residual Dense Network), adapted for image denoising (toy-first, pure torch).

    Original RDN is often used for super-resolution; here we keep resolution and predict a residual/noise map.
    Output is `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        num_blocks: int = 8,
        num_layers: int = 6,
        growth: int = 32,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        d = int(num_blocks)
        l = int(num_layers)
        g = int(growth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if d <= 0:
            raise ValueError("num_blocks must be > 0")

        self.sfe1 = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.sfe2 = nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True)

        self.rdbs = nn.ModuleList(
            [ResidualDenseBlock(f, growth=g, num_layers=l) for _ in range(d)]
        )

        self.gff_1x1 = nn.Conv2d(f * d, f, kernel_size=1, bias=True)
        self.gff_3x3 = nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True)

        self.out = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        f1 = F.relu(self.sfe1(x), inplace=True)
        f0 = F.relu(self.sfe2(f1), inplace=True)

        feats: list[torch.Tensor] = []
        y = f0
        for block in self.rdbs:
            y = block(y)
            feats.append(y)

        y = self.gff_3x3(F.relu(self.gff_1x1(torch.cat(feats, dim=1)), inplace=True))
        y = y + f1  # global residual

        residual = self.out(F.relu(y, inplace=True))
        return x - residual


_VARIANTS: dict[str, dict] = {
    "rdn_tiny": {"features": 32, "num_blocks": 3, "num_layers": 3, "growth": 16},
    "rdn_small": {"features": 48, "num_blocks": 6, "num_layers": 4, "growth": 24},
    "rdn_base": {"features": 64, "num_blocks": 10, "num_layers": 6, "growth": 32},
}


def build_rdn_denoiser(*, in_channels: int, variant: str = "rdn_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RDN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RDN(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        num_blocks=int(spec["num_blocks"]),
        num_layers=int(spec["num_layers"]),
        growth=int(spec["growth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_rdn_denoiser(in_channels=1, variant="rdn_tiny")
    y = m(noisy)
    print("rdn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

