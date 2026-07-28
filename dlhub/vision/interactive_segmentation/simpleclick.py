from __future__ import annotations
from ._common import build_baseline_inter, smoke_test_inter

_VARIANTS = {
    "simpleclick_tiny": {"width": 24, "depth": 1},
    "simpleclick_small": {"width": 32, "depth": 2},
    "simpleclick_base": {"width": 48, "depth": 3},
}


def build_simpleclick_interactive_segmenter(
    *, in_channels: int, variant: str = "simpleclick_small", width_mult: float = 1.0
):
    return build_baseline_inter(
        family="simpleclick",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_inter(build_simpleclick_interactive_segmenter, "simpleclick_tiny")
