
from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    'free_at_guard_tiny': {'width': 24, 'depth': 1},
    'free_at_guard_small': {'width': 32, 'depth': 2},
    'free_at_guard_base': {'width': 48, 'depth': 3},
}


def build_free_at_guard_robust_model(*, in_channels: int, variant: str = 'free_at_guard_small', width_mult: float = 1.0):
    return build_toy_vision_direction(
        family='free_at_guard',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == '__main__':
    smoke_test_direction(build_free_at_guard_robust_model, 'free_at_guard_tiny')
