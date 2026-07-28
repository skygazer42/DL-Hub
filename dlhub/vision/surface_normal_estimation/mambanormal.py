from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "mambanormal_tiny": {"width": 24, "depth": 1},
    "mambanormal_small": {"width": 32, "depth": 2},
    "mambanormal_base": {"width": 48, "depth": 3},
}


def build_mambanormal_normal_estimator(
    *, in_channels: int, variant: str = "mambanormal_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="mambanormal",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_mambanormal_normal_estimator, "mambanormal_tiny")
