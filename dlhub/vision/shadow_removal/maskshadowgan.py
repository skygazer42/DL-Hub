from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "maskshadowgan_tiny": {"width": 24, "depth": 1},
    "maskshadowgan_small": {"width": 32, "depth": 2},
    "maskshadowgan_base": {"width": 48, "depth": 3},
}


def build_maskshadowgan_shadow_remover(
    *, in_channels: int, variant: str = "maskshadowgan_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="maskshadowgan",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_maskshadowgan_shadow_remover, "maskshadowgan_tiny")
