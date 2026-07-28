from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "shape_symbol_tiny": {"width": 24, "depth": 1},
    "shape_symbol_small": {"width": 32, "depth": 2},
    "shape_symbol_base": {"width": 48, "depth": 3},
}


def build_shape_symbol_symbol_recognizer(
    *, in_channels: int, variant: str = "shape_symbol_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="shape_symbol",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_shape_symbol_symbol_recognizer, "shape_symbol_tiny")
