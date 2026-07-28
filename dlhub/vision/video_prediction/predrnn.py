from __future__ import annotations
from ._common import build_baseline_video_understander, smoke_test_vu

_VARIANTS = {
    "predrnn_tiny": {"width": 24, "depth": 1},
    "predrnn_small": {"width": 32, "depth": 2},
    "predrnn_base": {"width": 48, "depth": 3},
}


def build_predrnn_(
    *,
    in_channels: int,
    num_classes: int = 8,
    variant: str = "predrnn_small",
    width_mult: float = 1.0,
):
    return build_baseline_video_understander(
        family="predrnn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vu(build_predrnn_, "predrnn_tiny")
