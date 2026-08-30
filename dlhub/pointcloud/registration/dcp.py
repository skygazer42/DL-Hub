from __future__ import annotations
from ._common import build_compact_registration_model, validate_registration_model

_VARIANTS = {
    "dcp_tiny": {"width": 24, "depth": 1},
    "dcp_small": {"width": 32, "depth": 2},
    "dcp_base": {"width": 48, "depth": 3},
}


def build_dcp_registrar(*, variant: str = "dcp_small", width_mult: float = 1.0):
    return build_compact_registration_model(
        family="dcp", variants=_VARIANTS, variant=str(variant), width_mult=float(width_mult)
    )


if __name__ == "__main__":
    validate_registration_model(build_dcp_registrar, "dcp_tiny")
