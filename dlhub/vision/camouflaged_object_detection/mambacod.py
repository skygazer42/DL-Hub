from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "mambacod_tiny": {"width": 24, "depth": 1},
    "mambacod_small": {"width": 32, "depth": 2},
    "mambacod_base": {"width": 48, "depth": 3},
}


def build_mambacod_camouflaged_detector(
    *, in_channels: int, variant: str = "mambacod_small", width_mult: float = 1.0
):
    return build_toy_model(
        family="mambacod",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_mambacod_camouflaged_detector, "mambacod_tiny")
