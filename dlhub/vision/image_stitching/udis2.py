from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "udis2_tiny": {"width": 24, "depth": 1},
    "udis2_small": {"width": 32, "depth": 2},
    "udis2_base": {"width": 48, "depth": 3},
}


def build_udis2_stitcher(
    *, in_channels: int, variant: str = "udis2_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="udis2",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_udis2_stitcher, "udis2_tiny")
