from __future__ import annotations

from torch import nn

from ._common import build_scene_flow_estimator, smoke_test_scene_flow_estimator

_VARIANTS: dict[str, dict[str, int | float]] = {
    "flow3d_pointnet_tiny": {"width": 48, "depth": 2, "hidden_mult": 2, "refine_steps": 1, "delta_scale": 0.85},
    "flow3d_pointnet_small": {"width": 64, "depth": 3, "hidden_mult": 2, "refine_steps": 2, "delta_scale": 0.9},
    "flow3d_pointnet_base": {"width": 96, "depth": 4, "hidden_mult": 2, "refine_steps": 2, "delta_scale": 0.95},
}


def build_flow3d_pointnet_scene_flow_estimator(
    *,
    in_channels: int,
    variant: str = "flow3d_pointnet_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_scene_flow_estimator(
        family="flow3d_pointnet",
        in_channels=int(in_channels),
        variant=str(variant),
        variants=_VARIANTS,
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_scene_flow_estimator(build_flow3d_pointnet_scene_flow_estimator, "flow3d_pointnet_tiny")
