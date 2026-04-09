
from __future__ import annotations

from ._common import build_toy_splatter, smoke_test_splatter

_VARIANTS = {
    'compact_splat_tiny': {'width': 24, 'depth': 1},
    'compact_splat_small': {'width': 32, 'depth': 2},
    'compact_splat_base': {'width': 48, 'depth': 3},
}


def build_compact_splat_splatter(*, in_channels: int, variant: str = 'compact_splat_small', width_mult: float = 1.0):
    return build_toy_splatter(
        family='compact_splat',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == '__main__':
    smoke_test_splatter(build_compact_splat_splatter, 'compact_splat_tiny')
