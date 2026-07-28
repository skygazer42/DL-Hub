from __future__ import annotations
from ._common import build_baseline_counter, smoke_test_counter

_VARIANTS = {
    "gridcount_tiny": {"width": 24, "depth": 1},
    "gridcount_small": {"width": 32, "depth": 2},
    "gridcount_base": {"width": 48, "depth": 3},
}


def build_gridcount_(
    *, in_channels: int, variant: str = "gridcount_small", width_mult: float = 1.0
):
    return build_baseline_counter(
        family="gridcount",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_counter(build_gridcount_, "gridcount_tiny")
