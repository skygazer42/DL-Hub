from __future__ import annotations
from torch import nn
from ._common import build_baseline_depth_estimator, smoke_test_depth

_VARIANTS = {
    "manydepth_tiny": {"width": 24, "depth": 1},
    "manydepth_small": {"width": 32, "depth": 2},
    "manydepth_base": {"width": 48, "depth": 3},
}


def build_manydepth_depth_estimator(
    *, in_channels: int, variant: str = "manydepth_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_depth_estimator(
        family="manydepth",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        bins=False,
    )


if __name__ == "__main__":
    smoke_test_depth(build_manydepth_depth_estimator, "manydepth_tiny")
