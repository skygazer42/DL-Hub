from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "regionlayout_tiny": {"width": 24, "depth": 1},
    "regionlayout_small": {"width": 32, "depth": 2},
    "regionlayout_base": {"width": 48, "depth": 3},
}


def build_regionlayout_layout_analyzer(
    *, in_channels: int, variant: str = "regionlayout_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="regionlayout",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_regionlayout_layout_analyzer, "regionlayout_tiny")
