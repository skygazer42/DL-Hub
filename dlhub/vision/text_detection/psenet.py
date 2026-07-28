from __future__ import annotations
from ._common import build_baseline_text_detector, smoke_test_text

_VARIANTS = {
    "psenet_tiny": {"width": 24, "depth": 1},
    "psenet_small": {"width": 32, "depth": 2},
    "psenet_base": {"width": 48, "depth": 3},
}


def build_psenet_text_detector(
    *, in_channels: int, variant: str = "psenet_small", width_mult: float = 1.0
):
    return build_baseline_text_detector(
        family="psenet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text(build_psenet_text_detector, "psenet_tiny")
