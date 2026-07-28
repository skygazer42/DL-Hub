from __future__ import annotations
from ._common import build_baseline_inter, smoke_test_inter

_VARIANTS = {
    "ritm_tiny": {"width": 24, "depth": 1},
    "ritm_small": {"width": 32, "depth": 2},
    "ritm_base": {"width": 48, "depth": 3},
}


def build_ritm_interactive_segmenter(
    *, in_channels: int, variant: str = "ritm_small", width_mult: float = 1.0
):
    return build_baseline_inter(
        family="ritm",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_inter(build_ritm_interactive_segmenter, "ritm_tiny")
