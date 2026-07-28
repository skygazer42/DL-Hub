from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "paq2piq_tiny": {"width": 24, "depth": 1},
    "paq2piq_small": {"width": 32, "depth": 2},
    "paq2piq_base": {"width": 48, "depth": 3},
}


def build_paq2piq_iqa_model(
    *, in_channels: int, variant: str = "paq2piq_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="paq2piq",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_paq2piq_iqa_model, "paq2piq_tiny")
