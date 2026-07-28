from __future__ import annotations
from ._common import build_baseline_video_restorer, smoke_test_video

_VARIANTS = {
    "basicvsrpp_tiny": {"width": 24, "depth": 1},
    "basicvsrpp_small": {"width": 32, "depth": 2},
    "basicvsrpp_base": {"width": 48, "depth": 3},
}


def build_basicvsrpp_video_restorer(
    *, in_channels: int, variant: str = "basicvsrpp_small", width_mult: float = 1.0
):
    return build_baseline_video_restorer(
        family="basicvsrpp",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_video(build_basicvsrpp_video_restorer, "basicvsrpp_tiny")
