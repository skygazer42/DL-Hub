from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "semantic_contour_tiny": {"width": 24, "depth": 1},
    "semantic_contour_small": {"width": 32, "depth": 2},
    "semantic_contour_base": {"width": 48, "depth": 3},
}


def build_semantic_contour_contour_detector(
    *, in_channels: int, variant: str = "semantic_contour_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="semantic_contour",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_semantic_contour_contour_detector, "semantic_contour_tiny")
