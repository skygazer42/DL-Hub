from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "layoutformer_tiny": {"width": 24, "depth": 1},
    "layoutformer_small": {"width": 32, "depth": 2},
    "layoutformer_base": {"width": 48, "depth": 3},
}


def build_layoutformer_layout_analyzer(
    *, in_channels: int, variant: str = "layoutformer_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="layoutformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_layoutformer_layout_analyzer, "layoutformer_tiny")
