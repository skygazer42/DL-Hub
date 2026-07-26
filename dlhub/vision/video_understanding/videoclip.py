from __future__ import annotations
from ._common import build_toy_video_understander, smoke_test_vu

_VARIANTS = {
    "videoclip_tiny": {"width": 24, "depth": 1},
    "videoclip_small": {"width": 32, "depth": 2},
    "videoclip_base": {"width": 48, "depth": 3},
}


def build_videoclip_video_understander(
    *, in_channels: int, num_classes: int, variant: str = "videoclip_small", width_mult: float = 1.0
):
    return build_toy_video_understander(
        family="videoclip",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vu(build_videoclip_video_understander, "videoclip_tiny")
