from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "dorn_normal_tiny": {"width": 24, "depth": 1},
    "dorn_normal_small": {"width": 32, "depth": 2},
    "dorn_normal_base": {"width": 48, "depth": 3},
}


def build_dorn_normal_normal_estimator(
    *, in_channels: int, variant: str = "dorn_normal_small", width_mult: float = 1.0
):
    return build_toy_model(
        family="dorn_normal",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_dorn_normal_normal_estimator, "dorn_normal_tiny")
