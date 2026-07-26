from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "chem_symbol_tiny": {"width": 24, "depth": 1},
    "chem_symbol_small": {"width": 32, "depth": 2},
    "chem_symbol_base": {"width": 48, "depth": 3},
}


def build_chem_symbol_symbol_recognizer(
    *, in_channels: int, variant: str = "chem_symbol_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="chem_symbol",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_chem_symbol_symbol_recognizer, "chem_symbol_tiny")
