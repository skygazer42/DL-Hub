from __future__ import annotations
from ._common import build_compact_registration_model, validate_registration_model

_VARIANTS = {
    "regtr_tiny": {"width": 24, "depth": 1},
    "regtr_small": {"width": 32, "depth": 2},
    "regtr_base": {"width": 48, "depth": 3},
}


def build_regtr_registrar(*, variant: str = "regtr_small", width_mult: float = 1.0):
    return build_compact_registration_model(
        family="regtr", variants=_VARIANTS, variant=str(variant), width_mult=float(width_mult)
    )


if __name__ == "__main__":
    validate_registration_model(build_regtr_registrar, "regtr_tiny")
