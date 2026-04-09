
from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    'ifrnet_toy_tiny': {'width': 24, 'depth': 1},
    'ifrnet_toy_small': {'width': 32, 'depth': 2},
    'ifrnet_toy_base': {'width': 48, 'depth': 3},
}


def build_ifrnet_toy_interpolator(*, in_channels: int, variant: str = 'ifrnet_toy_small', width_mult: float = 1.0):
    return build_toy_vision_direction(
        family='ifrnet_toy',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == '__main__':
    smoke_test_direction(build_ifrnet_toy_interpolator, 'ifrnet_toy_tiny')
