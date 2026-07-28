from __future__ import annotations
from torch import nn
from ._common import build_baseline_rs_detector, smoke_test_rs

_VARIANTS = {
    "yolox_tiny": {"width": 24, "depth": 1},
    "yolox_small": {"width": 32, "depth": 2},
    "yolox_base": {"width": 48, "depth": 3},
}


def build_yolox_rs_detector(
    *, in_channels: int, num_classes: int, variant: str = "yolox_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_rs_detector(
        family="yolox",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        oriented=True,
    )


if __name__ == "__main__":
    smoke_test_rs(build_yolox_rs_detector, "yolox_tiny")
