from __future__ import annotations

from torch import nn

from ._common import build_scene_flow_estimator, smoke_test_scene_flow_estimator

_VARIANTS: dict[str, dict[str, int | float]] = {
    "diffusion_flow3d_tiny": {
        "width": 64,
        "depth": 5,
        "hidden_mult": 3,
        "refine_steps": 3,
        "delta_scale": 0.8,
    },
    "diffusion_flow3d_small": {
        "width": 96,
        "depth": 6,
        "hidden_mult": 3,
        "refine_steps": 4,
        "delta_scale": 0.85,
    },
    "diffusion_flow3d_base": {
        "width": 128,
        "depth": 7,
        "hidden_mult": 3,
        "refine_steps": 5,
        "delta_scale": 0.9,
    },
}


def build_diffusion_flow3d_scene_flow_estimator(
    *,
    in_channels: int,
    variant: str = "diffusion_flow3d_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_scene_flow_estimator(
        family="diffusion_flow3d",
        in_channels=int(in_channels),
        variant=str(variant),
        variants=_VARIANTS,
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_scene_flow_estimator(
        build_diffusion_flow3d_scene_flow_estimator, "diffusion_flow3d_tiny"
    )
