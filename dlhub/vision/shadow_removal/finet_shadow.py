from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "finet_shadow_tiny": {"width": 24, "depth": 1},
    "finet_shadow_small": {"width": 32, "depth": 2},
    "finet_shadow_base": {"width": 48, "depth": 3},
}


def build_finet_shadow_shadow_remover(
    *, in_channels: int, variant: str = "finet_shadow_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="finet_shadow",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_finet_shadow_shadow_remover, "finet_shadow_tiny")
