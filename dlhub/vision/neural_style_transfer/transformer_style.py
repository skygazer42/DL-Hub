from __future__ import annotations

from torch import nn

from ._common import build_toy_stylizer, smoke_test_stylizer


_VARIANTS: dict[str, dict[str, int]] = {'transformer_style_tiny': {'width': 24, 'depth': 1}, 'transformer_style_small': {'width': 36, 'depth': 2}, 'transformer_style_base': {'width': 48, 'depth': 3}}


def build_transformer_style_stylizer(
    *,
    in_channels: int,
    variant: str = 'transformer_style_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_stylizer(
        family='transformer_style',
        mode='transformer',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_stylizer(build_transformer_style_stylizer, 'transformer_style_tiny')
