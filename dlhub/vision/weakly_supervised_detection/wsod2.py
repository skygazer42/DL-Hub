from __future__ import annotations
from ._common import build_baseline_ws_detector, smoke_test_ws

_VARIANTS = {
    "wsod2_tiny": {"width": 24, "depth": 1},
    "wsod2_small": {"width": 32, "depth": 2},
    "wsod2_base": {"width": 48, "depth": 3},
}


def build_wsod2_ws_detector(
    *, in_channels: int, num_classes: int, variant: str = "wsod2_small", width_mult: float = 1.0
):
    return build_baseline_ws_detector(
        family="wsod2",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ws(build_wsod2_ws_detector, "wsod2_tiny")
