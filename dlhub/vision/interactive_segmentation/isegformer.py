from __future__ import annotations
from ._common import build_toy_inter, smoke_test_inter

_VARIANTS = {
    "isegformer_tiny": {"width": 24, "depth": 1},
    "isegformer_small": {"width": 32, "depth": 2},
    "isegformer_base": {"width": 48, "depth": 3},
}


def build_isegformer_interactive_segmenter(
    *, in_channels: int, variant: str = "isegformer_small", width_mult: float = 1.0
):
    return build_toy_inter(
        family="isegformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_inter(build_isegformer_interactive_segmenter, "isegformer_tiny")
