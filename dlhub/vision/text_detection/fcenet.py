from __future__ import annotations
from ._common import build_toy_text_detector, smoke_test_text

_VARIANTS = {
    "fcenet_tiny": {"width": 24, "depth": 1},
    "fcenet_small": {"width": 32, "depth": 2},
    "fcenet_base": {"width": 48, "depth": 3},
}


def build_fcenet_text_detector(
    *, in_channels: int, variant: str = "fcenet_small", width_mult: float = 1.0
):
    return build_toy_text_detector(
        family="fcenet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text(build_fcenet_text_detector, "fcenet_tiny")
