from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "tres_iqa_tiny": {"width": 24, "depth": 1},
    "tres_iqa_small": {"width": 32, "depth": 2},
    "tres_iqa_base": {"width": 48, "depth": 3},
}


def build_tres_iqa_iqa_model(
    *, in_channels: int, variant: str = "tres_iqa_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="tres_iqa",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_tres_iqa_iqa_model, "tres_iqa_tiny")
