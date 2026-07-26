from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    "adv_prop_guard_tiny": {"width": 24, "depth": 1},
    "adv_prop_guard_small": {"width": 32, "depth": 2},
    "adv_prop_guard_base": {"width": 48, "depth": 3},
}


def build_adv_prop_guard_robust_model(
    *, in_channels: int, variant: str = "adv_prop_guard_small", width_mult: float = 1.0
):
    return build_toy_vision_direction(
        family="adv_prop_guard",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_adv_prop_guard_robust_model, "adv_prop_guard_tiny")
