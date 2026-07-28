from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "quad_rectifier_tiny": {"width": 24, "depth": 1},
    "quad_rectifier_small": {"width": 32, "depth": 2},
    "quad_rectifier_base": {"width": 48, "depth": 3},
}


def build_quad_rectifier_dewarper(
    *, in_channels: int, variant: str = "quad_rectifier_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="quad_rectifier",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_quad_rectifier_dewarper, "quad_rectifier_tiny")
