from __future__ import annotations

from torch import nn

from ._common import build_toy_stylizer, smoke_test_stylizer


_VARIANTS: dict[str, dict[str, int]] = {'mamba_style_tiny': {'width': 24, 'depth': 1}, 'mamba_style_small': {'width': 36, 'depth': 2}, 'mamba_style_base': {'width': 48, 'depth': 3}}


def build_mamba_style_stylizer(
    *,
    in_channels: int,
    variant: str = 'mamba_style_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_stylizer(
        family='mamba_style',
        mode='mamba',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_stylizer(build_mamba_style_stylizer, 'mamba_style_tiny')
