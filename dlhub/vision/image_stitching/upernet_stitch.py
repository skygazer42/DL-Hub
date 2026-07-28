from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "upernet_stitch_tiny": {"width": 24, "depth": 1},
    "upernet_stitch_small": {"width": 32, "depth": 2},
    "upernet_stitch_base": {"width": 48, "depth": 3},
}


def build_upernet_stitch_stitcher(
    *, in_channels: int, variant: str = "upernet_stitch_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="upernet_stitch",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_upernet_stitch_stitcher, "upernet_stitch_tiny")
