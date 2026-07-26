from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "transtide_harm_tiny": {"width": 24, "depth": 1},
    "transtide_harm_small": {"width": 32, "depth": 2},
    "transtide_harm_base": {"width": 48, "depth": 3},
}


def build_transtide_harm_harmonizer(
    *, in_channels: int, variant: str = "transtide_harm_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="transtide_harm",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_transtide_harm_harmonizer, "transtide_harm_tiny")
