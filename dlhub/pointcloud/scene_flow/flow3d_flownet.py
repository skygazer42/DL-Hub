from __future__ import annotations

from torch import nn

from ._common import build_scene_flow_estimator, smoke_test_scene_flow_estimator

_VARIANTS: dict[str, dict[str, int | float]] = {
    "flow3d_flownet_tiny": {
        "width": 56,
        "depth": 2,
        "hidden_mult": 3,
        "refine_steps": 2,
        "delta_scale": 0.95,
    },
    "flow3d_flownet_small": {
        "width": 80,
        "depth": 3,
        "hidden_mult": 3,
        "refine_steps": 2,
        "delta_scale": 1.0,
    },
    "flow3d_flownet_base": {
        "width": 112,
        "depth": 4,
        "hidden_mult": 3,
        "refine_steps": 3,
        "delta_scale": 1.05,
    },
}


def build_flow3d_flownet_scene_flow_estimator(
    *,
    in_channels: int,
    variant: str = "flow3d_flownet_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_scene_flow_estimator(
        family="flow3d_flownet",
        in_channels=int(in_channels),
        variant=str(variant),
        variants=_VARIANTS,
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_scene_flow_estimator(
        build_flow3d_flownet_scene_flow_estimator, "flow3d_flownet_tiny"
    )
