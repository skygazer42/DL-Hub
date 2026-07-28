from __future__ import annotations
from torch import nn
from ._common import build_baseline_rs_detector, smoke_test_rs

_VARIANTS = {
    "faster_rcnn_tiny": {"width": 24, "depth": 1},
    "faster_rcnn_small": {"width": 32, "depth": 2},
    "faster_rcnn_base": {"width": 48, "depth": 3},
}


def build_faster_rcnn_rs_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "faster_rcnn_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_rs_detector(
        family="faster_rcnn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        oriented=True,
    )


if __name__ == "__main__":
    smoke_test_rs(build_faster_rcnn_rs_detector, "faster_rcnn_tiny")
