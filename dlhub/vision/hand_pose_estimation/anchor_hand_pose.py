from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_pose_estimator, smoke_test_hand_pose_estimator


_VARIANTS: dict[str, dict[str, int]] = {
    "anchor_hand_pose_tiny": {"width": 24, "depth": 1, "num_keypoints": 10},
    "anchor_hand_pose_small": {"width": 36, "depth": 2, "num_keypoints": 10},
    "anchor_hand_pose_base": {"width": 48, "depth": 3, "num_keypoints": 10},
}


def build_anchor_hand_pose_hand_pose_estimator(
    *,
    in_channels: int,
    variant: str = "anchor_hand_pose_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_pose_estimator(
        family="anchor_hand_pose",
        mode="anchor",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_pose_estimator(
        build_anchor_hand_pose_hand_pose_estimator, "anchor_hand_pose_tiny"
    )
