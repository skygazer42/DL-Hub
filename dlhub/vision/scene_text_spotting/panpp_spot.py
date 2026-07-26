from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "panpp_spot_tiny": {"width": 24, "depth": 1},
    "panpp_spot_small": {"width": 32, "depth": 2},
    "panpp_spot_base": {"width": 48, "depth": 3},
}


def build_panpp_spot_text_spotter(
    *, in_channels: int, variant: str = "panpp_spot_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="panpp_spot",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_panpp_spot_text_spotter, "panpp_spot_tiny")
