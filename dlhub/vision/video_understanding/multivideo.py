from __future__ import annotations
from ._common import build_baseline_video_understander, smoke_test_vu

_VARIANTS = {
    "multivideo_tiny": {"width": 24, "depth": 1},
    "multivideo_small": {"width": 32, "depth": 2},
    "multivideo_base": {"width": 48, "depth": 3},
}


def build_multivideo_video_understander(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "multivideo_small",
    width_mult: float = 1.0,
):
    return build_baseline_video_understander(
        family="multivideo",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vu(build_multivideo_video_understander, "multivideo_tiny")
