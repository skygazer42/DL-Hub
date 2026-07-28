from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "ransacnet_tiny": {"width": 24, "depth": 1},
    "ransacnet_small": {"width": 32, "depth": 2},
    "ransacnet_base": {"width": 48, "depth": 3},
}


def build_ransacnet_stitcher(
    *, in_channels: int, variant: str = "ransacnet_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="ransacnet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_ransacnet_stitcher, "ransacnet_tiny")
