from __future__ import annotations

from torch import nn

from ._common import build_scene_flow_estimator, smoke_test_scene_flow_estimator

_VARIANTS: dict[str, dict[str, int | float]] = {
    "prompt_flow3d_tiny": {
        "width": 56,
        "depth": 3,
        "hidden_mult": 3,
        "refine_steps": 2,
        "delta_scale": 0.9,
    },
    "prompt_flow3d_small": {
        "width": 88,
        "depth": 4,
        "hidden_mult": 3,
        "refine_steps": 3,
        "delta_scale": 0.95,
    },
    "prompt_flow3d_base": {
        "width": 120,
        "depth": 5,
        "hidden_mult": 3,
        "refine_steps": 4,
        "delta_scale": 1.0,
    },
}


def build_prompt_flow3d_scene_flow_estimator(
    *,
    in_channels: int,
    variant: str = "prompt_flow3d_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_scene_flow_estimator(
        family="prompt_flow3d",
        in_channels=int(in_channels),
        variant=str(variant),
        variants=_VARIANTS,
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_scene_flow_estimator(build_prompt_flow3d_scene_flow_estimator, "prompt_flow3d_tiny")
