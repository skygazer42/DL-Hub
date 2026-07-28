from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "agentformer_tiny": {"width": 24, "depth": 1},
    "agentformer_small": {"width": 32, "depth": 2},
    "agentformer_base": {"width": 48, "depth": 3},
}


def build_agentformer_trajectory_predictor(
    *,
    coord_dim: int,
    variant: str = "agentformer_small",
    width_mult: float = 1.0,
    pred_steps: int = 12,
):
    return build_baseline_model(
        family="agentformer",
        variants=_VARIANTS,
        coord_dim=int(coord_dim),
        variant=str(variant),
        width_mult=float(width_mult),
        pred_steps=int(pred_steps),
    )


if __name__ == "__main__":
    smoke_test_model(build_agentformer_trajectory_predictor, "agentformer_tiny")
