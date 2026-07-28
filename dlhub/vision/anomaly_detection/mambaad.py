from __future__ import annotations
from torch import nn
from ._common import build_baseline_anomaly_detector, smoke_test_anomaly

_VARIANTS = {
    "mambaad_tiny": {"width": 24, "depth": 1},
    "mambaad_small": {"width": 32, "depth": 2},
    "mambaad_base": {"width": 48, "depth": 3},
}


def build_mambaad_anomaly_detector(
    *, in_channels: int, variant: str = "mambaad_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_anomaly_detector(
        family="mambaad",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_anomaly(build_mambaad_anomaly_detector, "mambaad_tiny")
