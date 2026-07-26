from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    "amt_interp_tiny": {"width": 24, "depth": 1},
    "amt_interp_small": {"width": 32, "depth": 2},
    "amt_interp_base": {"width": 48, "depth": 3},
}


def build_amt_interp_interpolator(
    *, in_channels: int, variant: str = "amt_interp_small", width_mult: float = 1.0
):
    return build_toy_vision_direction(
        family="amt_interp",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_amt_interp_interpolator, "amt_interp_tiny")
