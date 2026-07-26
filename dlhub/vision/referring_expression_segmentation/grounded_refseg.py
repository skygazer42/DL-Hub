from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "grounded_refseg_tiny": {"width": 24, "depth": 1},
    "grounded_refseg_small": {"width": 32, "depth": 2},
    "grounded_refseg_base": {"width": 48, "depth": 3},
}


def build_grounded_refseg_refexp_segmenter(
    *, in_channels: int, variant: str = "grounded_refseg_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="grounded_refseg",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_grounded_refseg_refexp_segmenter, "grounded_refseg_tiny")
