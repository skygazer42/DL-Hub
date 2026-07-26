from __future__ import annotations
from ._common import build_toy_change, smoke_test_change

_VARIANTS = {
    "fcsiamdiff_tiny": {"width": 24, "depth": 1},
    "fcsiamdiff_small": {"width": 32, "depth": 2},
    "fcsiamdiff_base": {"width": 48, "depth": 3},
}


def build_fcsiamdiff_change_detector(
    *, in_channels: int, variant: str = "fcsiamdiff_small", width_mult: float = 1.0
):
    return build_toy_change(
        family="fcsiamdiff",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_change(build_fcsiamdiff_change_detector, "fcsiamdiff_tiny")
