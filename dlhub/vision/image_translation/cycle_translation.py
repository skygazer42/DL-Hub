from __future__ import annotations

from torch import nn

from ._common import build_toy_translator, smoke_test_translator


_VARIANTS: dict[str, dict[str, int]] = {'cycle_translation_tiny': {'width': 24, 'depth': 1}, 'cycle_translation_small': {'width': 36, 'depth': 2}, 'cycle_translation_base': {'width': 48, 'depth': 3}}


def build_cycle_translation_translator(
    *,
    in_channels: int,
    variant: str = 'cycle_translation_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_translator(
        family='cycle_translation',
        mode='cycle',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_translator(build_cycle_translation_translator, 'cycle_translation_tiny')
