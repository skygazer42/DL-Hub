from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "nnet_tiny": {"width": 24, "depth": 1},
    "nnet_small": {"width": 32, "depth": 2},
    "nnet_base": {"width": 48, "depth": 3},
}


def build_nnet_normal_estimator(
    *, in_channels: int, variant: str = "nnet_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="nnet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_nnet_normal_estimator, "nnet_tiny")
