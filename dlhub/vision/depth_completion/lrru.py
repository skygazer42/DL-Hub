from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "lrru_tiny": {"width": 24, "depth": 1},
    "lrru_small": {"width": 32, "depth": 2},
    "lrru_base": {"width": 48, "depth": 3},
}


def build_lrru_depth_completer(
    *, in_channels: int, variant: str = "lrru_small", width_mult: float = 1.0
):
    return build_toy_model(
        family="lrru",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_lrru_depth_completer, "lrru_tiny")
