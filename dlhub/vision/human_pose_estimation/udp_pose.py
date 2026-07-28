from __future__ import annotations
from ._common import build_baseline_pose, smoke_test_pose

_VARIANTS = {
    "udp_pose_tiny": {"width": 24, "depth": 1},
    "udp_pose_small": {"width": 32, "depth": 2},
    "udp_pose_base": {"width": 48, "depth": 3},
}


def build_udp_pose_pose_estimator(
    *, in_channels: int, num_joints: int, variant: str = "udp_pose_small", width_mult: float = 1.0
):
    return build_baseline_pose(
        family="udp_pose",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_joints=int(num_joints),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_pose(build_udp_pose_pose_estimator, "udp_pose_tiny")
