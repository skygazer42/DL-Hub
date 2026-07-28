from __future__ import annotations
from ._common import build_baseline_pose6d, smoke_test_6d

_VARIANTS = {
    "megapose_tiny": {"width": 24, "depth": 1},
    "megapose_small": {"width": 32, "depth": 2},
    "megapose_base": {"width": 48, "depth": 3},
}


def build_megapose_pose6d_estimator(
    *, in_channels: int, num_objects: int, variant: str = "megapose_small", width_mult: float = 1.0
):
    return build_baseline_pose6d(
        family="megapose",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_objects=int(num_objects),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_6d(build_megapose_pose6d_estimator, "megapose_tiny")
