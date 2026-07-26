from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "simplevqa_tiny": {"width": 24, "depth": 1},
    "simplevqa_small": {"width": 32, "depth": 2},
    "simplevqa_base": {"width": 48, "depth": 3},
}


def build_simplevqa_vqa_model(
    *, in_channels: int, variant: str = "simplevqa_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="simplevqa",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_simplevqa_vqa_model, "simplevqa_tiny")
