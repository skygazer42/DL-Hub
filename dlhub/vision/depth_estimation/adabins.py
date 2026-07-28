from __future__ import annotations
from torch import nn
from ._common import build_baseline_depth_estimator, smoke_test_depth

_VARIANTS = {
    "adabins_tiny": {"width": 24, "depth": 1},
    "adabins_small": {"width": 32, "depth": 2},
    "adabins_base": {"width": 48, "depth": 3},
}


def build_adabins_depth_estimator(
    *, in_channels: int, variant: str = "adabins_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_depth_estimator(
        family="adabins",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        bins=True,
    )


if __name__ == "__main__":
    smoke_test_depth(build_adabins_depth_estimator, "adabins_tiny")
