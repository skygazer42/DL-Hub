import torch
import torch.nn.functional as F
from torch import nn


class IRCNN(nn.Module):
    """IRCNN-style dilated residual CNN denoiser (compact-first, pure torch).

    IRCNN is often used as a restoration prior with dilated convolutions. This compact-first
    implementation:
    - uses a stack of dilated conv layers
    - predicts a residual/noise map and returns `x - residual`
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        dilations: tuple[int, ...] = (1, 2, 3, 4, 3, 2, 1),
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        dils = tuple(int(d) for d in dilations)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if len(dils) < 3:
            raise ValueError("dilations must have length >= 3")
        if any(d <= 0 for d in dils):
            raise ValueError("dilations must be positive ints")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(f, f, kernel_size=3, padding=di, dilation=di, bias=True) for di in dils]
        )
        self.out_conv = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = F.relu(self.in_conv(x), inplace=True)
        for conv in self.convs:
            y = F.relu(conv(y), inplace=True)
        residual = self.out_conv(y)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "ircnn_tiny": {"features": 32, "dilations": (1, 2, 2, 2, 2, 2, 1)},
    "ircnn_small": {"features": 48, "dilations": (1, 2, 3, 3, 3, 2, 1)},
    "ircnn_base": {"features": 64, "dilations": (1, 2, 3, 4, 3, 2, 1)},
}


def build_ircnn_denoiser(*, in_channels: int, variant: str = "ircnn_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown IRCNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return IRCNN(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        dilations=tuple(int(d) for d in spec["dilations"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_ircnn_denoiser(in_channels=1, variant="ircnn_tiny")
    y = m(noisy)
    print("ircnn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
