from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "gma_flow_tiny": {"width": 24, "depth": 1},
    "gma_flow_small": {"width": 32, "depth": 2},
    "gma_flow_base": {"width": 48, "depth": 3},
}


def build_gma_flow_flow_estimator(
    *, in_channels: int, variant: str = "gma_flow_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="gma_flow",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_gma_flow_flow_estimator, "gma_flow_tiny")
