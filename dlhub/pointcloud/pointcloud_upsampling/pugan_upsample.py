from __future__ import annotations
from torch import nn
from ._common import build_toy_upsampler, smoke_test_upsampler

_VARIANTS: dict[str, dict[str, int]] = {
    "pugan_upsample_tiny": {"width": 24, "depth": 1, "up_factor": 2},
    "pugan_upsample_small": {"width": 32, "depth": 2, "up_factor": 2},
    "pugan_upsample_base": {"width": 48, "depth": 3, "up_factor": 4},
}


def build_pugan_upsample_upsampler(
    *, in_channels: int, variant: str = "pugan_upsample_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_upsampler(
        family="pugan_upsample",
        mode="pugan",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_upsampler(build_pugan_upsample_upsampler, "pugan_upsample_tiny")
