from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "jcsod_tiny": {"width": 24, "depth": 1},
    "jcsod_small": {"width": 32, "depth": 2},
    "jcsod_base": {"width": 48, "depth": 3},
}


def build_jcsod_camouflaged_detector(
    *, in_channels: int, variant: str = "jcsod_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="jcsod",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_jcsod_camouflaged_detector, "jcsod_tiny")
