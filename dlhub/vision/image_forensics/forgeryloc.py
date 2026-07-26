from __future__ import annotations
from torch import nn
from ._common import build_toy_anomaly_detector, smoke_test_anomaly

_VARIANTS = {
    "forgeryloc_tiny": {"width": 24, "depth": 1},
    "forgeryloc_small": {"width": 32, "depth": 2},
    "forgeryloc_base": {"width": 48, "depth": 3},
}


def build_forgeryloc_(
    *, in_channels: int, variant: str = "forgeryloc_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_anomaly_detector(
        family="forgeryloc",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_anomaly(build_forgeryloc_, "forgeryloc_tiny")
