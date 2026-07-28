from __future__ import annotations
from ._common import build_baseline_video_enhancer, smoke_test_ve

_VARIANTS = {
    "rvrt_enhance_tiny": {"width": 24, "depth": 1},
    "rvrt_enhance_small": {"width": 32, "depth": 2},
    "rvrt_enhance_base": {"width": 48, "depth": 3},
}


def build_rvrt_enhance_video_enhancer(
    *, in_channels: int, variant: str = "rvrt_enhance_small", width_mult: float = 1.0
):
    return build_baseline_video_enhancer(
        family="rvrt_enhance",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ve(build_rvrt_enhance_video_enhancer, "rvrt_enhance_tiny")
