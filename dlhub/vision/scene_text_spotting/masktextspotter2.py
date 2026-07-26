from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "masktextspotter2_tiny": {"width": 24, "depth": 1},
    "masktextspotter2_small": {"width": 32, "depth": 2},
    "masktextspotter2_base": {"width": 48, "depth": 3},
}


def build_masktextspotter2_text_spotter(
    *, in_channels: int, variant: str = "masktextspotter2_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="masktextspotter2",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_masktextspotter2_text_spotter, "masktextspotter2_tiny")
