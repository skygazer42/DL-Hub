from __future__ import annotations

from torch import nn

from ._common import build_toy_deepfake_detector, smoke_test_deepfake_detector


_VARIANTS: dict[str, dict[str, int]] = {'region_deepfake_tiny': {'width': 24, 'depth': 1}, 'region_deepfake_small': {'width': 36, 'depth': 2}, 'region_deepfake_base': {'width': 48, 'depth': 3}}


def build_region_deepfake_deepfake_detector(
    *,
    in_channels: int,
    variant: str = 'region_deepfake_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_deepfake_detector(
        family='region_deepfake',
        mode='region',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_deepfake_detector(build_region_deepfake_deepfake_detector, 'region_deepfake_tiny')
