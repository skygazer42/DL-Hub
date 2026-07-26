from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "pecnet_tiny": {"width": 24, "depth": 1},
    "pecnet_small": {"width": 32, "depth": 2},
    "pecnet_base": {"width": 48, "depth": 3},
}


def build_pecnet_trajectory_predictor(
    *, coord_dim: int, variant: str = "pecnet_small", width_mult: float = 1.0, pred_steps: int = 12
):
    return build_toy_model(
        family="pecnet",
        variants=_VARIANTS,
        coord_dim=int(coord_dim),
        variant=str(variant),
        width_mult=float(width_mult),
        pred_steps=int(pred_steps),
    )


if __name__ == "__main__":
    smoke_test_model(build_pecnet_trajectory_predictor, "pecnet_tiny")
