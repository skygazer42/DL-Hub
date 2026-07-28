from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "super_slomo_tiny": {"width": 24, "depth": 1},
    "super_slomo_small": {"width": 32, "depth": 2},
    "super_slomo_base": {"width": 48, "depth": 3},
}


def build_super_slomo_interpolator(
    *, in_channels: int, variant: str = "super_slomo_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="super_slomo",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_super_slomo_interpolator, "super_slomo_tiny")
