from __future__ import annotations
from ._common import build_compact_registration_model, validate_registration_model

_VARIANTS = {
    "rpmnet_tiny": {"width": 24, "depth": 1},
    "rpmnet_small": {"width": 32, "depth": 2},
    "rpmnet_base": {"width": 48, "depth": 3},
}


def build_rpmnet_registrar(*, variant: str = "rpmnet_small", width_mult: float = 1.0):
    return build_compact_registration_model(
        family="rpmnet", variants=_VARIANTS, variant=str(variant), width_mult=float(width_mult)
    )


if __name__ == "__main__":
    validate_registration_model(build_rpmnet_registrar, "rpmnet_tiny")
