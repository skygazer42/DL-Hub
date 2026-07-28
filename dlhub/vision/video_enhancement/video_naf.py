from __future__ import annotations
from ._common import build_baseline_video_enhancer, smoke_test_ve

_VARIANTS = {
    "video_naf_tiny": {"width": 24, "depth": 1},
    "video_naf_small": {"width": 32, "depth": 2},
    "video_naf_base": {"width": 48, "depth": 3},
}


def build_video_naf_video_enhancer(
    *, in_channels: int, variant: str = "video_naf_small", width_mult: float = 1.0
):
    return build_baseline_video_enhancer(
        family="video_naf",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ve(build_video_naf_video_enhancer, "video_naf_tiny")
