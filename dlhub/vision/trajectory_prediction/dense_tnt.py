from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "dense_tnt_tiny": {"width": 24, "depth": 1},
    "dense_tnt_small": {"width": 32, "depth": 2},
    "dense_tnt_base": {"width": 48, "depth": 3},
}


def build_dense_tnt_trajectory_predictor(
    *,
    coord_dim: int,
    variant: str = "dense_tnt_small",
    width_mult: float = 1.0,
    pred_steps: int = 12,
):
    return build_toy_model(
        family="dense_tnt",
        variants=_VARIANTS,
        coord_dim=int(coord_dim),
        variant=str(variant),
        width_mult=float(width_mult),
        pred_steps=int(pred_steps),
    )


if __name__ == "__main__":
    smoke_test_model(build_dense_tnt_trajectory_predictor, "dense_tnt_tiny")
