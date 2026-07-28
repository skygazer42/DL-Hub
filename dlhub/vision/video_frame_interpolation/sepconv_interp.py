from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "sepconv_interp_tiny": {"width": 24, "depth": 1},
    "sepconv_interp_small": {"width": 32, "depth": 2},
    "sepconv_interp_base": {"width": 48, "depth": 3},
}


def build_sepconv_interp_interpolator(
    *, in_channels: int, variant: str = "sepconv_interp_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="sepconv_interp",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_sepconv_interp_interpolator, "sepconv_interp_tiny")
