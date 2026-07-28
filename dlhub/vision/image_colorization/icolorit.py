from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "icolorit_tiny": {"width": 24, "depth": 1},
    "icolorit_small": {"width": 32, "depth": 2},
    "icolorit_base": {"width": 48, "depth": 3},
}


def build_icolorit_colorizer(
    *, in_channels: int, variant: str = "icolorit_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="icolorit",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_icolorit_colorizer, "icolorit_tiny")
