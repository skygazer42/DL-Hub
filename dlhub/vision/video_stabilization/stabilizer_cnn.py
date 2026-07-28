from __future__ import annotations
from ._common import build_baseline_stabilizer, smoke_test_stabilizer

_VARIANTS = {
    "stabilizer_cnn_tiny": {"width": 24, "depth": 1},
    "stabilizer_cnn_small": {"width": 32, "depth": 2},
    "stabilizer_cnn_base": {"width": 48, "depth": 3},
}


def build_stabilizer_cnn_stabilizer(
    *, in_channels: int, variant: str = "stabilizer_cnn_small", width_mult: float = 1.0
):
    return build_baseline_stabilizer(
        family="stabilizer_cnn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_stabilizer(build_stabilizer_cnn_stabilizer, "stabilizer_cnn_tiny")
