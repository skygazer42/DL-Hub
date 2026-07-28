"""RCDNet (Rain Convolutional Dictionary Network) - compact-first implementation.

Reference (original idea):
- "Recurrent Convolutional Dictionary Network for Single Image Deraining"

Compact interpretation:
- Alternate between rain-code estimation and clean-image update.
- Keep updates lightweight and shape-preserving for synthetic denoising tracks.
"""

import torch
import torch.nn.functional as F
from torch import nn


class _UpdateBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(x + y, inplace=True)


class RCDNet(nn.Module):
    """Compact RCDNet-style derainer with unrolled recurrent updates."""

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 24,
        stages: int = 4,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        t = int(stages)
        b = int(blocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if t <= 0:
            raise ValueError("stages must be > 0")
        if b <= 0:
            raise ValueError("blocks must be > 0")

        self.in_channels = c_in
        self.stages = t
        self.enc = nn.Conv2d(c_in * 2, f, kernel_size=3, padding=1, bias=True)
        self.rain_body = nn.Sequential(*[_UpdateBlock(f) for _ in range(b)])
        self.clean_body = nn.Sequential(*[_UpdateBlock(f) for _ in range(b)])
        self.rain_head = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)
        self.clean_head = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {int(x.shape[1])} channels"
            )

        y = x
        rain = torch.zeros_like(x)
        for _ in range(int(self.stages)):
            feat = F.relu(self.enc(torch.cat([y, rain], dim=1)), inplace=True)
            rain = self.rain_head(self.rain_body(feat))
            clean_delta = self.clean_head(self.clean_body(feat))
            y = x - rain + clean_delta * 0.1
        return y


_VARIANTS: dict[str, dict] = {
    "rcdnet_tiny": {"features": 16, "stages": 3, "blocks": 1},
    "rcdnet_small": {"features": 24, "stages": 4, "blocks": 2},
    "rcdnet_base": {"features": 32, "stages": 6, "blocks": 3},
}


def build_rcdnet_denoiser(*, in_channels: int, variant: str = "rcdnet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RCDNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RCDNet(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        stages=int(spec["stages"]),
        blocks=int(spec["blocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_rcdnet_denoiser(in_channels=3, variant="rcdnet_tiny")
    y = m(x)
    print("rcdnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
