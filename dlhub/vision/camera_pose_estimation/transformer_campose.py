from __future__ import annotations

from torch import nn

from ._common import build_toy_camera_pose_estimator, smoke_test_camera_pose_estimator


_VARIANTS: dict[str, dict[str, int]] = {"transformer_campose_tiny": {"width": 24, "depth": 1}, "transformer_campose_small": {"width": 36, "depth": 2}, "transformer_campose_base": {"width": 48, "depth": 3}}


def build_transformer_campose_camera_pose_estimator(*, in_channels: int, variant: str = "transformer_campose_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_camera_pose_estimator(family="transformer_campose", mode="transformer", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_camera_pose_estimator(build_transformer_campose_camera_pose_estimator, "transformer_campose_tiny")
