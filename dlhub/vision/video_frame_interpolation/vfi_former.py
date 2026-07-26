from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    "vfi_former_tiny": {"width": 24, "depth": 1},
    "vfi_former_small": {"width": 32, "depth": 2},
    "vfi_former_base": {"width": 48, "depth": 3},
}


def build_vfi_former_interpolator(
    *, in_channels: int, variant: str = "vfi_former_small", width_mult: float = 1.0
):
    return build_toy_vision_direction(
        family="vfi_former",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_vfi_former_interpolator, "vfi_former_tiny")
