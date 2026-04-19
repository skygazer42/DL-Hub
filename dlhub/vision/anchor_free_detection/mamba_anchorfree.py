from __future__ import annotations

from torch import nn

from ._common import build_toy_detector, smoke_test_detector


_VARIANTS: dict[str, dict[str, int]] = {'mamba_anchorfree_tiny': {'width': 24, 'depth': 1, 'queries': 20}, 'mamba_anchorfree_small': {'width': 36, 'depth': 2, 'queries': 24}, 'mamba_anchorfree_base': {'width': 48, 'depth': 3, 'queries': 32}}


def build_mamba_anchorfree_detector(
    *,
    in_channels: int,
    variant: str = 'mamba_anchorfree_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_detector(
        family='mamba_anchorfree',
        mode='mamba',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_detector(build_mamba_anchorfree_detector, 'mamba_anchorfree_tiny')
