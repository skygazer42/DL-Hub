from __future__ import annotations
from ._common import build_baseline_inter, smoke_test_inter

_VARIANTS = {
    "dextr2_tiny": {"width": 24, "depth": 1},
    "dextr2_small": {"width": 32, "depth": 2},
    "dextr2_base": {"width": 48, "depth": 3},
}


def build_dextr2_interactive_segmenter(
    *, in_channels: int, variant: str = "dextr2_small", width_mult: float = 1.0
):
    return build_baseline_inter(
        family="dextr2",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_inter(build_dextr2_interactive_segmenter, "dextr2_tiny")
