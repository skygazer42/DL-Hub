from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import PixelShuffleUpsampler, _default_variants, check_low_res_image, validate_upscale_factor


class FSRCNN(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        shrink_channels: int,
        num_mapping_blocks: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        hidden = int(hidden_channels)
        shrink = int(shrink_channels)
        depth = int(num_mapping_blocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if hidden <= 0 or shrink <= 0:
            raise ValueError("hidden_channels and shrink_channels must be > 0")
        if depth <= 0:
            raise ValueError("num_mapping_blocks must be > 0")

        self.extract = nn.Conv2d(c_in, hidden, kernel_size=5, padding=2, bias=True)
        self.shrink = nn.Conv2d(hidden, shrink, kernel_size=1, bias=True)
        self.mapping = nn.ModuleList(
            nn.Conv2d(shrink, shrink, kernel_size=3, padding=1, bias=True) for _ in range(depth)
        )
        self.expand = nn.Conv2d(shrink, hidden, kernel_size=1, bias=True)
        self.upsample = PixelShuffleUpsampler(hidden, upscale_factor=2)
        self.reconstruct = nn.Conv2d(hidden, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, low_res: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_low_res_image(low_res)
        x = F.relu(self.extract(x), inplace=True)
        x = F.relu(self.shrink(x), inplace=True)
        for layer in self.mapping:
            x = F.relu(layer(x), inplace=True)
        x = F.relu(self.expand(x), inplace=True)
        x = F.relu(self.upsample(x), inplace=True)
        sr = self.reconstruct(x)
        return {"sr": sr}


_VARIANTS: dict[str, dict[str, int]] = _default_variants("fsrcnn")


def build_fsrcnn_super_resolver(
    *,
    in_channels: int,
    variant: str = "fsrcnn_small",
    upscale_factor: int = 2,
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del dropout
    validate_upscale_factor(upscale_factor)

    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FSRCNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = max(8, int(spec["width"] * float(width_mult)))
    shrink = max(4, width // 2)
    return FSRCNN(
        in_channels=int(in_channels),
        hidden_channels=width,
        shrink_channels=shrink,
        num_mapping_blocks=int(spec["depth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_fsrcnn_super_resolver(in_channels=3, variant="fsrcnn_tiny")
    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    print("fsrcnn_tiny", tuple(y["sr"].shape))
