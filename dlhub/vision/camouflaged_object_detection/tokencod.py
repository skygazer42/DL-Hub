from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "tokencod_tiny": {"width": 24, "depth": 1},
    "tokencod_small": {"width": 32, "depth": 2},
    "tokencod_base": {"width": 48, "depth": 3},
}


def build_tokencod_camouflaged_detector(
    *, in_channels: int, variant: str = "tokencod_small", width_mult: float = 1.0
):
    return build_toy_model(
        family="tokencod",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_tokencod_camouflaged_detector, "tokencod_tiny")
