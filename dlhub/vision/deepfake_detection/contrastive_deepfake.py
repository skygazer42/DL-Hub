from __future__ import annotations

from torch import nn

from ._common import build_toy_deepfake_detector, smoke_test_deepfake_detector


_VARIANTS: dict[str, dict[str, int]] = {'contrastive_deepfake_tiny': {'width': 24, 'depth': 1}, 'contrastive_deepfake_small': {'width': 36, 'depth': 2}, 'contrastive_deepfake_base': {'width': 48, 'depth': 3}}


def build_contrastive_deepfake_deepfake_detector(
    *,
    in_channels: int,
    variant: str = 'contrastive_deepfake_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_deepfake_detector(
        family='contrastive_deepfake',
        mode='contrastive',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_deepfake_detector(build_contrastive_deepfake_deepfake_detector, 'contrastive_deepfake_tiny')
