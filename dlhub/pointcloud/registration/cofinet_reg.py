from __future__ import annotations
from ._common import build_compact_registration_model, validate_registration_model

_VARIANTS = {
    "cofinet_reg_tiny": {"width": 24, "depth": 1},
    "cofinet_reg_small": {"width": 32, "depth": 2},
    "cofinet_reg_base": {"width": 48, "depth": 3},
}


def build_cofinet_reg_registrar(*, variant: str = "cofinet_reg_small", width_mult: float = 1.0):
    return build_compact_registration_model(
        family="cofinet_reg", variants=_VARIANTS, variant=str(variant), width_mult=float(width_mult)
    )


if __name__ == "__main__":
    validate_registration_model(build_cofinet_reg_registrar, "cofinet_reg_tiny")
