from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        r = int(reduction)
        hidden = max(8, c // max(1, r))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(self.pool(x))
        return x * w


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.ca = ChannelAttention(c, reduction=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.act(self.conv1(x)))
        y = self.ca(y)
        return x + y


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, *, num_blocks: int) -> None:
        super().__init__()
        c = int(channels)
        nb = int(num_blocks)
        if nb <= 0:
            raise ValueError("num_blocks must be > 0")
        self.body = nn.Sequential(*[RCAB(c) for _ in range(nb)])
        self.tail = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tail(self.body(x))
        return x + y


class RCAN(nn.Module):
    """RCAN-style blind denoiser (toy-first, pure torch).

    RCAN is typically used for super-resolution/restoration; in denoising we keep the same idea
    (deep residual groups with channel attention) and predict a residual that is subtracted.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 48,
        num_groups: int = 3,
        blocks_per_group: int = 4,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        ng = int(num_groups)
        nb = int(blocks_per_group)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if ng <= 0:
            raise ValueError("num_groups must be > 0")
        if nb <= 0:
            raise ValueError("blocks_per_group must be > 0")

        self.intro = nn.Conv2d(c_in, w0, kernel_size=3, padding=1, bias=True)
        self.groups = nn.Sequential(*[ResidualGroup(w0, num_blocks=nb) for _ in range(ng)])
        self.body_tail = nn.Conv2d(w0, w0, kernel_size=3, padding=1, bias=True)
        self.outro = nn.Conv2d(w0, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        f = self.intro(x)
        f = f + self.body_tail(self.groups(f))
        residual = self.outro(f)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "rcan_tiny": {"width": 32, "groups": 2, "blocks": 3},
    "rcan_small": {"width": 48, "groups": 3, "blocks": 4},
    "rcan_base": {"width": 64, "groups": 5, "blocks": 6},
}


def build_rcan_denoiser(
    *,
    in_channels: int,
    variant: str = "rcan_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RCAN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RCAN(
        in_channels=int(in_channels),
        width=int(spec["width"]),
        num_groups=int(spec["groups"]),
        blocks_per_group=int(spec["blocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_rcan_denoiser(in_channels=1, variant="rcan_tiny")
    y = m(noisy)
    print("rcan_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

