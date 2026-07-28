from __future__ import annotations
from torch import nn
from ._common import build_baseline_anomaly_detector, smoke_test_anomaly

_VARIANTS = {
    "patchcore_tiny": {"width": 24, "depth": 1},
    "patchcore_small": {"width": 32, "depth": 2},
    "patchcore_base": {"width": 48, "depth": 3},
}


def build_patchcore_anomaly_detector(
    *, in_channels: int, variant: str = "patchcore_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_anomaly_detector(
        family="patchcore",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_anomaly(build_patchcore_anomaly_detector, "patchcore_tiny")
