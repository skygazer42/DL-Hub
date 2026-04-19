from __future__ import annotations

from torch import nn

from ._common import build_toy_face_pose_estimator, smoke_test_face_pose_estimator


_VARIANTS: dict[str, dict[str, int]] = {'uncertainty_face_pose_tiny': {'width': 24, 'depth': 1}, 'uncertainty_face_pose_small': {'width': 36, 'depth': 2}, 'uncertainty_face_pose_base': {'width': 48, 'depth': 3}}


def build_uncertainty_face_pose_face_pose_estimator(
    *,
    in_channels: int,
    variant: str = 'uncertainty_face_pose_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_pose_estimator(
        family='uncertainty_face_pose',
        mode='uncertainty',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_pose_estimator(build_uncertainty_face_pose_face_pose_estimator, 'uncertainty_face_pose_tiny')
