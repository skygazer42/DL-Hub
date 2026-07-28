from __future__ import annotations
from ._common import build_baseline_text_detector, smoke_test_text

_VARIANTS = {
    "pan_tiny": {"width": 24, "depth": 1},
    "pan_small": {"width": 32, "depth": 2},
    "pan_base": {"width": 48, "depth": 3},
}


def build_pan_text_detector(
    *, in_channels: int, variant: str = "pan_small", width_mult: float = 1.0
):
    return build_baseline_text_detector(
        family="pan",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text(build_pan_text_detector, "pan_tiny")
