from __future__ import annotations

from torch import nn

from ._common import build_scene_flow_estimator, smoke_test_scene_flow_estimator

_VARIANTS: dict[str, dict[str, int | float]] = {
    "transformer_flow3d_tiny": {"width": 64, "depth": 4, "hidden_mult": 4, "refine_steps": 2, "delta_scale": 0.9},
    "transformer_flow3d_small": {"width": 96, "depth": 5, "hidden_mult": 4, "refine_steps": 3, "delta_scale": 0.95},
    "transformer_flow3d_base": {"width": 128, "depth": 6, "hidden_mult": 4, "refine_steps": 3, "delta_scale": 1.0},
}


def build_transformer_flow3d_scene_flow_estimator(
    *,
    in_channels: int,
    variant: str = "transformer_flow3d_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_scene_flow_estimator(
        family="transformer_flow3d",
        in_channels=int(in_channels),
        variant=str(variant),
        variants=_VARIANTS,
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_scene_flow_estimator(
        build_transformer_flow3d_scene_flow_estimator,
        "transformer_flow3d_tiny",
    )
