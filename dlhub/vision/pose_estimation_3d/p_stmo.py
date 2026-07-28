from __future__ import annotations
from ._common import build_baseline_pose3d, smoke_test_pose3d

_VARIANTS = {
    "p_stmo_tiny": {"width": 32, "depth": 1},
    "p_stmo_small": {"width": 48, "depth": 2},
    "p_stmo_base": {"width": 64, "depth": 3},
}


def build_p_stmo_pose3d_estimator(
    *, num_joints: int, variant: str = "p_stmo_small", width_mult: float = 1.0
):
    return build_baseline_pose3d(
        family="p_stmo",
        variants=_VARIANTS,
        num_joints=int(num_joints),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_pose3d(build_p_stmo_pose3d_estimator, "p_stmo_tiny")
