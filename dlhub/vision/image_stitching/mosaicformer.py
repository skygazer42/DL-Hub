from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "mosaicformer_tiny": {"width": 24, "depth": 1},
    "mosaicformer_small": {"width": 32, "depth": 2},
    "mosaicformer_base": {"width": 48, "depth": 3},
}


def build_mosaicformer_stitcher(
    *, in_channels: int, variant: str = "mosaicformer_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="mosaicformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_mosaicformer_stitcher, "mosaicformer_tiny")
