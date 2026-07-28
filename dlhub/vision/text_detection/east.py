from __future__ import annotations
from ._common import build_baseline_text_detector, smoke_test_text

_VARIANTS = {
    "east_tiny": {"width": 24, "depth": 1},
    "east_small": {"width": 32, "depth": 2},
    "east_base": {"width": 48, "depth": 3},
}


def build_east_text_detector(
    *, in_channels: int, variant: str = "east_small", width_mult: float = 1.0
):
    return build_baseline_text_detector(
        family="east",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text(build_east_text_detector, "east_tiny")
