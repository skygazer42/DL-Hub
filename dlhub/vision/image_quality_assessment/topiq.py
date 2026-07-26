from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "topiq_tiny": {"width": 24, "depth": 1},
    "topiq_small": {"width": 32, "depth": 2},
    "topiq_base": {"width": 48, "depth": 3},
}


def build_topiq_iqa_model(
    *, in_channels: int, variant: str = "topiq_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="topiq",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_topiq_iqa_model, "topiq_tiny")
