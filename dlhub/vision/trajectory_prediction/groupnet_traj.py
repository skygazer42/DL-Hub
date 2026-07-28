from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "groupnet_traj_tiny": {"width": 24, "depth": 1},
    "groupnet_traj_small": {"width": 32, "depth": 2},
    "groupnet_traj_base": {"width": 48, "depth": 3},
}


def build_groupnet_traj_trajectory_predictor(
    *,
    coord_dim: int,
    variant: str = "groupnet_traj_small",
    width_mult: float = 1.0,
    pred_steps: int = 12,
):
    return build_baseline_model(
        family="groupnet_traj",
        variants=_VARIANTS,
        coord_dim=int(coord_dim),
        variant=str(variant),
        width_mult=float(width_mult),
        pred_steps=int(pred_steps),
    )


if __name__ == "__main__":
    smoke_test_model(build_groupnet_traj_trajectory_predictor, "groupnet_traj_tiny")
