from __future__ import annotations

from torch import nn

from ._common import build_baseline_camera_pose_estimator, smoke_test_camera_pose_estimator


_VARIANTS: dict[str, dict[str, int]] = {
    "uncertainty_campose_tiny": {"width": 24, "depth": 1},
    "uncertainty_campose_small": {"width": 36, "depth": 2},
    "uncertainty_campose_base": {"width": 48, "depth": 3},
}


def build_uncertainty_campose_camera_pose_estimator(
    *, in_channels: int, variant: str = "uncertainty_campose_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_camera_pose_estimator(
        family="uncertainty_campose",
        mode="uncertainty",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_camera_pose_estimator(
        build_uncertainty_campose_camera_pose_estimator, "uncertainty_campose_tiny"
    )
