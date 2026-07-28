"""RESCAN (Recurrent SE Context Aggregation Net) - compact-first implementation.

Reference (original idea):
- "Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining"
  (ECCV 2018)

Compact interpretation:
- A recurrent residual predictor with SE-gated feature blocks.
- Iteratively subtract predicted rain residual from the current estimate.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _act() -> nn.Module:
    return nn.ReLU(inplace=True)


class SEBlock(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        c = int(channels)
        r = int(reduction)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if r <= 0:
            raise ValueError("reduction must be > 0")
        hidden = max(4, c // r)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(c, hidden, kernel_size=1, bias=True)
        self.act = _act()
        self.fc2 = nn.Conv2d(hidden, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = self.fc2(self.act(self.fc1(s)))
        return x * torch.sigmoid(s)


class ResSEBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.act = _act()
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.se = SEBlock(c, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.act(self.conv1(x)))
        y = self.se(y)
        return x + y


class RESCAN(nn.Module):
    """Compact RESCAN-style derainer.

    Iteratively updates `y` by subtracting predicted residuals. We keep weights shared
    to mimic the "recurrent" spirit while remaining small and training-friendly.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 24,
        blocks: int = 2,
        stages: int = 4,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        b = int(blocks)
        t = int(stages)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if b <= 0:
            raise ValueError("blocks must be > 0")
        if t <= 0:
            raise ValueError("stages must be > 0")

        self.intro = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(*[ResSEBlock(f) for _ in range(b)])
        self.head = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)
        self.stages = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = x
        for _ in range(int(self.stages)):
            feat = self.body(F.relu(self.intro(y), inplace=True))
            residual = self.head(feat)
            y = y - residual
        return y


_VARIANTS: dict[str, dict] = {
    "rescan_tiny": {"features": 16, "blocks": 1, "stages": 3},
    "rescan_small": {"features": 24, "blocks": 2, "stages": 4},
    "rescan_base": {"features": 32, "blocks": 3, "stages": 5},
}


def build_rescan_denoiser(
    *,
    in_channels: int,
    variant: str = "rescan_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RESCAN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RESCAN(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        blocks=int(spec["blocks"]),
        stages=int(spec["stages"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_rescan_denoiser(in_channels=1, variant="rescan_tiny")
    y = m(noisy)
    print("rescan_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
