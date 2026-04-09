
from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    'book_flatten_tiny': {'width': 24, 'depth': 1},
    'book_flatten_small': {'width': 32, 'depth': 2},
    'book_flatten_base': {'width': 48, 'depth': 3},
}


def build_book_flatten_dewarper(*, in_channels: int, variant: str = 'book_flatten_small', width_mult: float = 1.0):
    return build_toy_vision_direction(
        family='book_flatten',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == '__main__':
    smoke_test_direction(build_book_flatten_dewarper, 'book_flatten_tiny')
