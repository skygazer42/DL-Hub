from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "trackdet_head_tiny": {"width": 24, "depth": 1},
    "trackdet_head_small": {"width": 32, "depth": 2},
    "trackdet_head_base": {"width": 48, "depth": 3},
}


def build_trackdet_head_video_detector(
    *, in_channels: int, variant: str = "trackdet_head_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="trackdet_head",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_trackdet_head_video_detector, "trackdet_head_tiny")
