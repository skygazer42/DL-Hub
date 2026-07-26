from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "pwcnet_tiny": {"width": 24, "depth": 1},
    "pwcnet_small": {"width": 32, "depth": 2},
    "pwcnet_base": {"width": 48, "depth": 3},
}


def build_pwcnet_flow_estimator(
    *, in_channels: int, variant: str = "pwcnet_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="pwcnet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_pwcnet_flow_estimator, "pwcnet_tiny")
