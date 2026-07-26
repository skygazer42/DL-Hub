from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    "dff_det_tiny": {"width": 24, "depth": 1},
    "dff_det_small": {"width": 32, "depth": 2},
    "dff_det_base": {"width": 48, "depth": 3},
}


def build_dff_det_video_detector(
    *, in_channels: int, variant: str = "dff_det_small", width_mult: float = 1.0
):
    return build_toy_vision_direction(
        family="dff_det",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_dff_det_video_detector, "dff_det_tiny")
