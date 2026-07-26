from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    "flowrcnn_vid_tiny": {"width": 24, "depth": 1},
    "flowrcnn_vid_small": {"width": 32, "depth": 2},
    "flowrcnn_vid_base": {"width": 48, "depth": 3},
}


def build_flowrcnn_vid_video_detector(
    *, in_channels: int, variant: str = "flowrcnn_vid_small", width_mult: float = 1.0
):
    return build_toy_vision_direction(
        family="flowrcnn_vid",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_flowrcnn_vid_video_detector, "flowrcnn_vid_tiny")
