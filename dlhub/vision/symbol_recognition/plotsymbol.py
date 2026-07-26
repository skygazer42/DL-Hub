from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "plotsymbol_tiny": {"width": 24, "depth": 1},
    "plotsymbol_small": {"width": 32, "depth": 2},
    "plotsymbol_base": {"width": 48, "depth": 3},
}


def build_plotsymbol_symbol_recognizer(
    *, in_channels: int, variant: str = "plotsymbol_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="plotsymbol",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_plotsymbol_symbol_recognizer, "plotsymbol_tiny")
