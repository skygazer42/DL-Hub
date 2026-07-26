from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "anodet_tiny": {"width": 24, "depth": 1},
    "anodet_small": {"width": 32, "depth": 2},
    "anodet_base": {"width": 48, "depth": 3},
}


def build_anodet_defect_detector(
    *, in_channels: int, variant: str = "anodet_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="anodet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_anodet_defect_detector, "anodet_tiny")
