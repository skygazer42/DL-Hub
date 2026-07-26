from __future__ import annotations
from ._common import build_toy_pose, smoke_test_pose

_VARIANTS = {
    "temporalpose_track_tiny": {"width": 24, "depth": 1},
    "temporalpose_track_small": {"width": 32, "depth": 2},
    "temporalpose_track_base": {"width": 48, "depth": 3},
}


def build_temporalpose_track_(
    *,
    in_channels: int,
    num_joints: int,
    variant: str = "temporalpose_track_small",
    width_mult: float = 1.0,
):
    return build_toy_pose(
        family="temporalpose_track",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_joints=int(num_joints),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_pose(build_temporalpose_track_, "temporalpose_track_tiny")
