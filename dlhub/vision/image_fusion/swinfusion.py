from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "swinfusion_tiny": {"width": 24, "depth": 1},
    "swinfusion_small": {"width": 32, "depth": 2},
    "swinfusion_base": {"width": 48, "depth": 3},
}


def build_swinfusion_fuser(
    *, in_channels: int, variant: str = "swinfusion_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="swinfusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_swinfusion_fuser, "swinfusion_tiny")
