from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    "certified_guard_tiny": {"width": 24, "depth": 1},
    "certified_guard_small": {"width": 32, "depth": 2},
    "certified_guard_base": {"width": 48, "depth": 3},
}


def build_certified_guard_robust_model(
    *, in_channels: int, variant: str = "certified_guard_small", width_mult: float = 1.0
):
    return build_toy_vision_direction(
        family="certified_guard",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_certified_guard_robust_model, "certified_guard_tiny")
