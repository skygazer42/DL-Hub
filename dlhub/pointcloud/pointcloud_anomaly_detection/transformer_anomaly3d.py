from __future__ import annotations

from torch import nn

from ._common import build_toy_anomaly_detector, smoke_test_anomaly_detector

_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_anomaly3d_tiny": {"width": 24, "depth": 1},
    "transformer_anomaly3d_small": {"width": 32, "depth": 2},
    "transformer_anomaly3d_base": {"width": 48, "depth": 3},
}


def build_transformer_anomaly3d_anomaly_detector(
    *, in_channels: int, variant: str = "transformer_anomaly3d_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_anomaly_detector(
        family="transformer_anomaly3d",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_anomaly_detector(
        build_transformer_anomaly3d_anomaly_detector, "transformer_anomaly3d_tiny"
    )
