from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "guided_depth_tiny": {"width": 24, "depth": 1},
    "guided_depth_small": {"width": 32, "depth": 2},
    "guided_depth_base": {"width": 48, "depth": 3},
}


def build_guided_depth_depth_completer(
    *, in_channels: int, variant: str = "guided_depth_small", width_mult: float = 1.0
):
    return build_toy_model(
        family="guided_depth",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_guided_depth_depth_completer, "guided_depth_tiny")
