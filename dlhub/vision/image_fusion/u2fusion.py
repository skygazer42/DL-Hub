from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "u2fusion_tiny": {"width": 24, "depth": 1},
    "u2fusion_small": {"width": 32, "depth": 2},
    "u2fusion_base": {"width": 48, "depth": 3},
}


def build_u2fusion_fuser(
    *, in_channels: int, variant: str = "u2fusion_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="u2fusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_u2fusion_fuser, "u2fusion_tiny")
