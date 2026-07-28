"""DDN (Deep Detail Network) - compact-first implementation.

Reference (original idea):
- "Removing Rain from Single Images via a Deep Detail Network" (CVPR 2017)

Compact interpretation:
- Use a shallow dilated-convolution stack to estimate rain residual details.
- Return derained output as `x - residual`.
"""

import torch
import torch.nn.functional as F
from torch import nn


class _DilatedResBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__()
        c = int(channels)
        d = int(dilation)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if d <= 0:
            raise ValueError("dilation must be > 0")
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=d, dilation=d, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=d, dilation=d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(x + y, inplace=True)


class DDN(nn.Module):
    """Compact DDN-style derainer with shape-preserving residual prediction."""

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 24,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        b = int(blocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if b <= 0:
            raise ValueError("blocks must be > 0")

        self.in_channels = c_in
        self.stem = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        dilations = [1, 2, 4, 1]
        layers: list[nn.Module] = []
        for i in range(b):
            layers.append(_DilatedResBlock(f, dilation=dilations[i % len(dilations)]))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {int(x.shape[1])} channels"
            )

        feat = F.relu(self.stem(x), inplace=True)
        feat = self.body(feat)
        residual = self.head(feat)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "ddn_tiny": {"features": 16, "blocks": 2},
    "ddn_small": {"features": 24, "blocks": 4},
    "ddn_base": {"features": 32, "blocks": 6},
}


def build_ddn_denoiser(*, in_channels: int, variant: str = "ddn_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DDN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DDN(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        blocks=int(spec["blocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_ddn_denoiser(in_channels=1, variant="ddn_tiny")
    y = m(noisy)
    print("ddn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
