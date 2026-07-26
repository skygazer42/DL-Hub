from __future__ import annotations
from torch import nn
from ._common import build_toy_depth_estimator, smoke_test_depth

_VARIANTS = {
    "packnet_tiny": {"width": 24, "depth": 1},
    "packnet_small": {"width": 32, "depth": 2},
    "packnet_base": {"width": 48, "depth": 3},
}


def build_packnet_depth_estimator(
    *, in_channels: int, variant: str = "packnet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_depth_estimator(
        family="packnet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        bins=False,
    )


if __name__ == "__main__":
    smoke_test_depth(build_packnet_depth_estimator, "packnet_tiny")
