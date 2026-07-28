from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "predator_reg_tiny": {"width": 24, "depth": 1},
    "predator_reg_small": {"width": 32, "depth": 2},
    "predator_reg_base": {"width": 48, "depth": 3},
}


def build_predator_reg_registrar(*, variant: str = "predator_reg_small", width_mult: float = 1.0):
    return build_baseline_model(
        family="predator_reg",
        variants=_VARIANTS,
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_predator_reg_registrar, "predator_reg_tiny")
