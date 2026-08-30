from __future__ import annotations
from ._common import build_compact_registration_model, validate_registration_model

_VARIANTS = {
    "geoformer_reg_tiny": {"width": 24, "depth": 1},
    "geoformer_reg_small": {"width": 32, "depth": 2},
    "geoformer_reg_base": {"width": 48, "depth": 3},
}


def build_geoformer_reg_registrar(*, variant: str = "geoformer_reg_small", width_mult: float = 1.0):
    return build_compact_registration_model(
        family="geoformer_reg",
        variants=_VARIANTS,
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    validate_registration_model(build_geoformer_reg_registrar, "geoformer_reg_tiny")
