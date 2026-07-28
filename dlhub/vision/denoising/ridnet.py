import torch
from torch import nn


class CALayer(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        hidden = max(8, c // r)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class RIDBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.ca = CALayer(c, reduction=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        y = self.ca(y)
        return x + y


class RIDNetDenoiser(nn.Module):
    """RIDNet-style denoiser (compact-first, pure torch).

    A residual-in-residual CNN with channel attention blocks.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        n = int(num_blocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f <= 0:
            raise ValueError("features must be > 0")
        if n <= 0:
            raise ValueError("num_blocks must be > 0")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.blocks = nn.Sequential(*[RIDBlock(f) for _ in range(n)])
        self.out_conv = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        f = self.blocks(self.in_conv(x))
        return x + self.out_conv(f)


_VARIANTS: dict[str, dict] = {
    "ridnet_tiny": {"features": 32, "blocks": 3},
    "ridnet_small": {"features": 48, "blocks": 4},
    "ridnet_base": {"features": 64, "blocks": 6},
}


def build_ridnet_denoiser(
    *,
    in_channels: int,
    variant: str = "ridnet_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RIDNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RIDNetDenoiser(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        num_blocks=int(spec["blocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_ridnet_denoiser(in_channels=1, variant="ridnet_tiny")
    y = m(noisy)
    print("ridnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
