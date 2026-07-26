from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "bmn2_tiny": {"width": 24, "depth": 1},
    "bmn2_small": {"width": 32, "depth": 2},
    "bmn2_base": {"width": 48, "depth": 3},
}


def build_bmn2_tal_model(
    *, in_channels: int, variant: str = "bmn2_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="bmn2",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_bmn2_tal_model, "bmn2_tiny")
