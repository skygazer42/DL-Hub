from __future__ import annotations

from torch import nn

from ._common import build_scene_flow_estimator, smoke_test_scene_flow_estimator

_VARIANTS: dict[str, dict[str, int | float]] = {
    "raft3d_points_tiny": {"width": 64, "depth": 4, "hidden_mult": 2, "refine_steps": 3, "delta_scale": 1.0},
    "raft3d_points_small": {"width": 96, "depth": 5, "hidden_mult": 2, "refine_steps": 4, "delta_scale": 1.05},
    "raft3d_points_base": {"width": 128, "depth": 6, "hidden_mult": 2, "refine_steps": 4, "delta_scale": 1.1},
}


def build_raft3d_points_scene_flow_estimator(
    *,
    in_channels: int,
    variant: str = "raft3d_points_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_scene_flow_estimator(
        family="raft3d_points",
        in_channels=int(in_channels),
        variant=str(variant),
        variants=_VARIANTS,
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_scene_flow_estimator(build_raft3d_points_scene_flow_estimator, "raft3d_points_tiny")
