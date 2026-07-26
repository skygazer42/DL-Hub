from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    "flavr_toy_tiny": {"width": 24, "depth": 1},
    "flavr_toy_small": {"width": 32, "depth": 2},
    "flavr_toy_base": {"width": 48, "depth": 3},
}


def build_flavr_toy_interpolator(
    *, in_channels: int, variant: str = "flavr_toy_small", width_mult: float = 1.0
):
    return build_toy_vision_direction(
        family="flavr_toy",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_flavr_toy_interpolator, "flavr_toy_tiny")
