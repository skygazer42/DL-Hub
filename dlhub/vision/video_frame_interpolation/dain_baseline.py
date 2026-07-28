from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "dain_baseline_tiny": {"width": 24, "depth": 1},
    "dain_baseline_small": {"width": 32, "depth": 2},
    "dain_baseline_base": {"width": 48, "depth": 3},
}


def build_dain_baseline_interpolator(
    *, in_channels: int, variant: str = "dain_baseline_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="dain_baseline",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_dain_baseline_interpolator, "dain_baseline_tiny")
