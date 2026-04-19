from __future__ import annotations

from torch import nn

from ._common import build_toy_stylizer, smoke_test_stylizer


_VARIANTS: dict[str, dict[str, int]] = {'pix2pix_style_tiny': {'width': 24, 'depth': 1}, 'pix2pix_style_small': {'width': 36, 'depth': 2}, 'pix2pix_style_base': {'width': 48, 'depth': 3}}


def build_pix2pix_style_stylizer(
    *,
    in_channels: int,
    variant: str = 'pix2pix_style_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_stylizer(
        family='pix2pix_style',
        mode='pix2pix',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_stylizer(build_pix2pix_style_stylizer, 'pix2pix_style_tiny')
