from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "spatialwarp_tiny": {"width": 24, "depth": 1},
    "spatialwarp_small": {"width": 32, "depth": 2},
    "spatialwarp_base": {"width": 48, "depth": 3},
}


def build_spatialwarp_stitcher(
    *, in_channels: int, variant: str = "spatialwarp_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="spatialwarp",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_spatialwarp_stitcher, "spatialwarp_tiny")
