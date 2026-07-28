from __future__ import annotations
from torch import nn
from ._common import build_baseline_restoration, smoke_test_restoration

_VARIANTS = {
    "deblurganv2_tiny": {"width": 24, "depth": 1},
    "deblurganv2_small": {"width": 32, "depth": 2},
    "deblurganv2_base": {"width": 48, "depth": 3},
}


def build_deblurganv2_deblurrer(
    *, in_channels: int, variant: str = "deblurganv2_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_restoration(
        family="deblurganv2",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        out_key="deblurred",
    )


if __name__ == "__main__":
    smoke_test_restoration(build_deblurganv2_deblurrer, "deblurganv2_tiny", "deblurred")
