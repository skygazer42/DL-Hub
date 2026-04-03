"""SPANet (Spatial Attentive Network) - toy-first implementation.

Reference (original idea):
- "Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset" (CVPR 2019)

Toy interpretation:
- Predict a spatial rain-attention map and a rain residual map from shared features.
- Apply attention to residual and subtract from input.
"""

import torch
import torch.nn.functional as F
from torch import nn


class _ResBlock(nn.Module):
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


class SPANet(nn.Module):
    """Toy SPANet-style derainer with explicit spatial attention."""

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 24,
        blocks: int = 3,
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
        self.body = nn.Sequential(*[_ResBlock(f) for _ in range(b)])
        self.attn_head = nn.Conv2d(f, 1, kernel_size=3, padding=1, bias=True)
        self.residual_head = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

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
        attn = torch.sigmoid(self.attn_head(feat))
        residual = self.residual_head(feat)
        return x - residual * attn


_VARIANTS: dict[str, dict] = {
    "spanet_tiny": {"features": 16, "blocks": 2},
    "spanet_small": {"features": 24, "blocks": 3},
    "spanet_base": {"features": 32, "blocks": 5},
}


def build_spanet_denoiser(*, in_channels: int, variant: str = "spanet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SPANet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SPANet(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        blocks=int(spec["blocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_spanet_denoiser(in_channels=1, variant="spanet_tiny")
    y = m(noisy)
    print("spanet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
