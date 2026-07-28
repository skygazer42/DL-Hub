from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "graphfitb_tiny": {"width": 24, "depth": 1},
    "graphfitb_small": {"width": 32, "depth": 2},
    "graphfitb_base": {"width": 48, "depth": 3},
}


def build_graphfitb_fashion_compat_model(
    *, in_channels: int, variant: str = "graphfitb_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="graphfitb",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_graphfitb_fashion_compat_model, "graphfitb_tiny")
