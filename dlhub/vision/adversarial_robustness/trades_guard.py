from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "trades_guard_tiny": {"width": 24, "depth": 1},
    "trades_guard_small": {"width": 32, "depth": 2},
    "trades_guard_base": {"width": 48, "depth": 3},
}


def build_trades_guard_robust_model(
    *, in_channels: int, variant: str = "trades_guard_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="trades_guard",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_trades_guard_robust_model, "trades_guard_tiny")
