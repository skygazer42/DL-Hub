from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_regressor, smoke_test_hand_regressor


_VARIANTS: dict[str, dict[str, int]] = {'direct_finger_curvature_tiny': {'width': 24, 'depth': 1}, 'direct_finger_curvature_small': {'width': 36, 'depth': 2}, 'direct_finger_curvature_base': {'width': 48, 'depth': 3}}


def build_direct_finger_curvature_finger_curvature_estimator(
    *,
    in_channels: int,
    variant: str = 'direct_finger_curvature_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_regressor(
        family='direct_finger_curvature',
        mode='direct',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_regressor(build_direct_finger_curvature_finger_curvature_estimator, 'direct_finger_curvature_tiny')
