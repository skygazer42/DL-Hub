from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import (
    PixelShuffleUpsampler,
    ResidualBlock,
    check_low_res_image,
    validate_upscale_factor,
)


class CARNSR(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        features: int,
        num_blocks: int,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        depth = int(num_blocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if depth <= 0:
            raise ValueError("num_blocks must be > 0")

        self.head = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(
            *[ResidualBlock(f, res_scale=float(res_scale)) for _ in range(depth)]
        )
        self.body_tail = nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True)
        self.upsample = PixelShuffleUpsampler(f, upscale_factor=2)
        self.tail = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, low_res: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_low_res_image(low_res)
        f0 = self.head(x)
        y = self.body_tail(self.body(f0)) + f0
        y = F.gelu(self.upsample(y))
        sr = self.tail(y)
        return {"sr": sr}


_VARIANTS: dict[str, dict[str, float | int]] = {
    "carn_sr_tiny": {"features": 24, "num_blocks": 3, "res_scale": 0.1},
    "carn_sr_small": {"features": 32, "num_blocks": 5, "res_scale": 0.1},
    "carn_sr_base": {"features": 48, "num_blocks": 8, "res_scale": 0.1},
}


def build_carn_sr_super_resolver(
    *,
    in_channels: int,
    variant: str = "carn_sr_small",
    upscale_factor: int = 2,
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del dropout
    validate_upscale_factor(upscale_factor)

    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EDSR-SR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    features = max(8, int(int(spec["features"]) * float(width_mult)))
    return CARNSR(
        in_channels=int(in_channels),
        features=features,
        num_blocks=int(spec["num_blocks"]),
        res_scale=float(spec["res_scale"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_carn_sr_super_resolver(in_channels=3, variant="carn_sr_tiny")
    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    print("carn_sr_tiny", tuple(y["sr"].shape))
