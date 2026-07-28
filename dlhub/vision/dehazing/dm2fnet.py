from __future__ import annotations
from torch import nn
from ._common import build_baseline_restoration, smoke_test_restoration

_VARIANTS = {
    "dm2fnet_tiny": {"width": 24, "depth": 1},
    "dm2fnet_small": {"width": 32, "depth": 2},
    "dm2fnet_base": {"width": 48, "depth": 3},
}


def build_dm2fnet_dehazer(
    *, in_channels: int, variant: str = "dm2fnet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_restoration(
        family="dm2fnet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        out_key="dehazed",
    )


if __name__ == "__main__":
    smoke_test_restoration(build_dm2fnet_dehazer, "dm2fnet_tiny", "dehazed")
