from __future__ import annotations

from torch import nn

from ._common import build_baseline_camera_pose_estimator, smoke_test_camera_pose_estimator


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_campose_tiny": {"width": 24, "depth": 1},
    "prompt_campose_small": {"width": 36, "depth": 2},
    "prompt_campose_base": {"width": 48, "depth": 3},
}


def build_prompt_campose_camera_pose_estimator(
    *, in_channels: int, variant: str = "prompt_campose_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_camera_pose_estimator(
        family="prompt_campose",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_camera_pose_estimator(
        build_prompt_campose_camera_pose_estimator, "prompt_campose_tiny"
    )
