from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "s2d_completion_tiny": {"width": 24, "depth": 1},
    "s2d_completion_small": {"width": 32, "depth": 2},
    "s2d_completion_base": {"width": 48, "depth": 3},
}


def build_s2d_completion_depth_completer(
    *, in_channels: int, variant: str = "s2d_completion_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="s2d_completion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_s2d_completion_depth_completer, "s2d_completion_tiny")
