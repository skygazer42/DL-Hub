from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import (
    ChannelAttention,
    PixelShuffleUpsampler,
    check_low_res_image,
    validate_upscale_factor,
)


class RCABlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.attn = ChannelAttention(c, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.conv1(x))
        y = self.attn(self.conv2(y))
        return x + y


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, *, num_blocks: int) -> None:
        super().__init__()
        depth = int(num_blocks)
        if depth <= 0:
            raise ValueError("num_blocks must be > 0")
        c = int(channels)
        self.body = nn.Sequential(*[RCABlock(c) for _ in range(depth)])
        self.tail = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.tail(self.body(x))


class RCANSR(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        features: int,
        num_groups: int,
        blocks_per_group: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        groups = int(num_groups)
        blocks = int(blocks_per_group)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if groups <= 0 or blocks <= 0:
            raise ValueError("num_groups and blocks_per_group must be > 0")

        self.head = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.groups = nn.Sequential(*[ResidualGroup(f, num_blocks=blocks) for _ in range(groups)])
        self.body_tail = nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True)
        self.upsample = PixelShuffleUpsampler(f, upscale_factor=2)
        self.tail = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, low_res: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_low_res_image(low_res)
        f0 = self.head(x)
        y = self.body_tail(self.groups(f0)) + f0
        y = F.gelu(self.upsample(y))
        sr = self.tail(y)
        return {"sr": sr}


_VARIANTS: dict[str, dict[str, int]] = {
    "rcan_sr_tiny": {"features": 24, "groups": 2, "blocks": 2},
    "rcan_sr_small": {"features": 32, "groups": 3, "blocks": 3},
    "rcan_sr_base": {"features": 48, "groups": 4, "blocks": 4},
}


def build_rcan_sr_super_resolver(
    *,
    in_channels: int,
    variant: str = "rcan_sr_small",
    upscale_factor: int = 2,
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del dropout
    validate_upscale_factor(upscale_factor)

    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RCAN-SR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    features = max(8, int(int(spec["features"]) * float(width_mult)))
    return RCANSR(
        in_channels=int(in_channels),
        features=features,
        num_groups=int(spec["groups"]),
        blocks_per_group=int(spec["blocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_rcan_sr_super_resolver(in_channels=3, variant="rcan_sr_tiny")
    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    print("rcan_sr_tiny", tuple(y["sr"].shape))
