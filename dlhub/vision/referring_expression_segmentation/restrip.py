from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "restrip_tiny": {"width": 24, "depth": 1},
    "restrip_small": {"width": 32, "depth": 2},
    "restrip_base": {"width": 48, "depth": 3},
}


def build_restrip_refexp_segmenter(
    *, in_channels: int, variant: str = "restrip_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="restrip",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_restrip_refexp_segmenter, "restrip_tiny")
