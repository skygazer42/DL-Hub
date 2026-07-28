from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "l2cs_tiny": {"width": 24, "depth": 1},
    "l2cs_small": {"width": 32, "depth": 2},
    "l2cs_base": {"width": 48, "depth": 3},
}


def build_l2cs_gaze_estimator(
    *, in_channels: int, variant: str = "l2cs_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="l2cs",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_l2cs_gaze_estimator, "l2cs_tiny")
