
from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    'mamba_layout_gen_tiny': {'width': 24, 'depth': 1},
    'mamba_layout_gen_small': {'width': 32, 'depth': 2},
    'mamba_layout_gen_base': {'width': 48, 'depth': 3},
}


def build_mamba_layout_gen_layout_generator(*, in_channels: int, variant: str = 'mamba_layout_gen_small', width_mult: float = 1.0):
    return build_toy_vision_direction(
        family='mamba_layout_gen',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == '__main__':
    smoke_test_direction(build_mamba_layout_gen_layout_generator, 'mamba_layout_gen_tiny')
