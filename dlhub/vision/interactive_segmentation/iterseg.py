from __future__ import annotations
from ._common import build_toy_inter, smoke_test_inter

_VARIANTS = {
    "iterseg_tiny": {"width": 24, "depth": 1},
    "iterseg_small": {"width": 32, "depth": 2},
    "iterseg_base": {"width": 48, "depth": 3},
}


def build_iterseg_interactive_segmenter(
    *, in_channels: int, variant: str = "iterseg_small", width_mult: float = 1.0
):
    return build_toy_inter(
        family="iterseg",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_inter(build_iterseg_interactive_segmenter, "iterseg_tiny")
