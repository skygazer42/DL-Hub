from __future__ import annotations
from ._common import build_baseline_counter, smoke_test_counter

_VARIANTS = {
    "bl_tiny": {"width": 24, "depth": 1},
    "bl_small": {"width": 32, "depth": 2},
    "bl_base": {"width": 48, "depth": 3},
}


def build_bl_crowd_counter(*, in_channels: int, variant: str = "bl_small", width_mult: float = 1.0):
    return build_baseline_counter(
        family="bl",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_counter(build_bl_crowd_counter, "bl_tiny")
