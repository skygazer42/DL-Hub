from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "lite_aes_tiny": {"width": 24, "depth": 1},
    "lite_aes_small": {"width": 32, "depth": 2},
    "lite_aes_base": {"width": 48, "depth": 3},
}


def build_lite_aes_aesthetic_model(
    *, in_channels: int, variant: str = "lite_aes_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="lite_aes",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_lite_aes_aesthetic_model, "lite_aes_tiny")
