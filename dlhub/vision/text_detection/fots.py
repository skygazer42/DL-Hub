from __future__ import annotations
from ._common import build_toy_text_detector, smoke_test_text

_VARIANTS = {
    "fots_tiny": {"width": 24, "depth": 1},
    "fots_small": {"width": 32, "depth": 2},
    "fots_base": {"width": 48, "depth": 3},
}


def build_fots_text_detector(
    *, in_channels: int, variant: str = "fots_small", width_mult: float = 1.0
):
    return build_toy_text_detector(
        family="fots",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text(build_fots_text_detector, "fots_tiny")
