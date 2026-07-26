from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import _default_variants, check_low_res_image, validate_upscale_factor


class OmniSR(nn.Module):
    def __init__(self, *, in_channels: int, hidden_channels: int, bottleneck_channels: int) -> None:
        super().__init__()
        c_in = int(in_channels)
        hidden = int(hidden_channels)
        bottleneck = int(bottleneck_channels)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if hidden <= 0 or bottleneck <= 0:
            raise ValueError("hidden_channels and bottleneck_channels must be > 0")

        self.head = nn.Conv2d(c_in, hidden, kernel_size=9, padding=4, bias=True)
        self.body = nn.Conv2d(hidden, bottleneck, kernel_size=5, padding=2, bias=True)
        self.tail = nn.Conv2d(bottleneck, c_in, kernel_size=5, padding=2, bias=True)

    def forward(self, low_res: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_low_res_image(low_res)
        x = F.interpolate(x, scale_factor=2.0, mode="bicubic", align_corners=False)
        x = F.relu(self.head(x), inplace=True)
        x = F.relu(self.body(x), inplace=True)
        sr = self.tail(x)
        return {"sr": sr}


_VARIANTS: dict[str, dict[str, int]] = _default_variants("omnisr")


def build_omnisr_super_resolver(
    *,
    in_channels: int,
    variant: str = "omnisr_small",
    upscale_factor: int = 2,
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del dropout
    validate_upscale_factor(upscale_factor)

    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown OmniSR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = max(8, int(spec["width"] * float(width_mult)))
    bottleneck = max(4, width // 2)
    return OmniSR(
        in_channels=int(in_channels),
        hidden_channels=width,
        bottleneck_channels=bottleneck,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_omnisr_super_resolver(in_channels=3, variant="omnisr_tiny")
    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    print("omnisr_tiny", tuple(y["sr"].shape))

