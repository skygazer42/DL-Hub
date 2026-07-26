from __future__ import annotations
from ._common import build_toy_pose6d, smoke_test_6d

_VARIANTS = {
    "ffb6d_tiny": {"width": 24, "depth": 1},
    "ffb6d_small": {"width": 32, "depth": 2},
    "ffb6d_base": {"width": 48, "depth": 3},
}


def build_ffb6d_pose6d_estimator(
    *, in_channels: int, num_objects: int, variant: str = "ffb6d_small", width_mult: float = 1.0
):
    return build_toy_pose6d(
        family="ffb6d",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_objects=int(num_objects),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_6d(build_ffb6d_pose6d_estimator, "ffb6d_tiny")
