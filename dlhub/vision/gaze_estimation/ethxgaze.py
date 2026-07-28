from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "ethxgaze_tiny": {"width": 24, "depth": 1},
    "ethxgaze_small": {"width": 32, "depth": 2},
    "ethxgaze_base": {"width": 48, "depth": 3},
}


def build_ethxgaze_gaze_estimator(
    *, in_channels: int, variant: str = "ethxgaze_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="ethxgaze",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_ethxgaze_gaze_estimator, "ethxgaze_tiny")
