from __future__ import annotations
from ._common import build_baseline_pose3d, smoke_test_pose3d

_VARIANTS = {
    "mixste_tiny": {"width": 32, "depth": 1},
    "mixste_small": {"width": 48, "depth": 2},
    "mixste_base": {"width": 64, "depth": 3},
}


def build_mixste_pose3d_estimator(
    *, num_joints: int, variant: str = "mixste_small", width_mult: float = 1.0
):
    return build_baseline_pose3d(
        family="mixste",
        variants=_VARIANTS,
        num_joints=int(num_joints),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_pose3d(build_mixste_pose3d_estimator, "mixste_tiny")
