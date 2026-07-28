from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "ema_vfi_tiny": {"width": 24, "depth": 1},
    "ema_vfi_small": {"width": 32, "depth": 2},
    "ema_vfi_base": {"width": 48, "depth": 3},
}


def build_ema_vfi_interpolator(
    *, in_channels: int, variant: str = "ema_vfi_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="ema_vfi",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_ema_vfi_interpolator, "ema_vfi_tiny")
