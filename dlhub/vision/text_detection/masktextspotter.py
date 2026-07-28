from __future__ import annotations
from ._common import build_baseline_text_detector, smoke_test_text

_VARIANTS = {
    "masktextspotter_tiny": {"width": 24, "depth": 1},
    "masktextspotter_small": {"width": 32, "depth": 2},
    "masktextspotter_base": {"width": 48, "depth": 3},
}


def build_masktextspotter_text_detector(
    *, in_channels: int, variant: str = "masktextspotter_small", width_mult: float = 1.0
):
    return build_baseline_text_detector(
        family="masktextspotter",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text(build_masktextspotter_text_detector, "masktextspotter_tiny")
