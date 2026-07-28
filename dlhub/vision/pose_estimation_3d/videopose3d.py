from __future__ import annotations
from ._common import build_baseline_pose3d, smoke_test_pose3d

_VARIANTS = {
    "videopose3d_tiny": {"width": 32, "depth": 1},
    "videopose3d_small": {"width": 48, "depth": 2},
    "videopose3d_base": {"width": 64, "depth": 3},
}


def build_videopose3d_pose3d_estimator(
    *, num_joints: int, variant: str = "videopose3d_small", width_mult: float = 1.0
):
    return build_baseline_pose3d(
        family="videopose3d",
        variants=_VARIANTS,
        num_joints=int(num_joints),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_pose3d(build_videopose3d_pose3d_estimator, "videopose3d_tiny")
