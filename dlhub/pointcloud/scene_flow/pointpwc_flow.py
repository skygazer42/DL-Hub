from __future__ import annotations

from torch import nn

from ._common import build_scene_flow_estimator, smoke_test_scene_flow_estimator

_VARIANTS: dict[str, dict[str, int | float]] = {
    "pointpwc_flow_tiny": {"width": 48, "depth": 3, "hidden_mult": 2, "refine_steps": 2, "delta_scale": 0.9},
    "pointpwc_flow_small": {"width": 72, "depth": 4, "hidden_mult": 2, "refine_steps": 3, "delta_scale": 0.95},
    "pointpwc_flow_base": {"width": 96, "depth": 5, "hidden_mult": 2, "refine_steps": 3, "delta_scale": 1.0},
}


def build_pointpwc_flow_scene_flow_estimator(
    *,
    in_channels: int,
    variant: str = "pointpwc_flow_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_scene_flow_estimator(
        family="pointpwc_flow",
        in_channels=int(in_channels),
        variant=str(variant),
        variants=_VARIANTS,
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_scene_flow_estimator(build_pointpwc_flow_scene_flow_estimator, "pointpwc_flow_tiny")
