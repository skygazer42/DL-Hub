from __future__ import annotations
from ._common import build_toy_pose3d, smoke_test_pose3d

_VARIANTS = {
    "gast_net_tiny": {"width": 32, "depth": 1},
    "gast_net_small": {"width": 48, "depth": 2},
    "gast_net_base": {"width": 64, "depth": 3},
}


def build_gast_net_pose3d_estimator(
    *, num_joints: int, variant: str = "gast_net_small", width_mult: float = 1.0
):
    return build_toy_pose3d(
        family="gast_net",
        variants=_VARIANTS,
        num_joints=int(num_joints),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_pose3d(build_gast_net_pose3d_estimator, "gast_net_tiny")
