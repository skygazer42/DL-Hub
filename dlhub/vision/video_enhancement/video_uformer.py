from __future__ import annotations
from ._common import build_baseline_video_enhancer, smoke_test_ve

_VARIANTS = {
    "video_uformer_tiny": {"width": 24, "depth": 1},
    "video_uformer_small": {"width": 32, "depth": 2},
    "video_uformer_base": {"width": 48, "depth": 3},
}


def build_video_uformer_video_enhancer(
    *, in_channels: int, variant: str = "video_uformer_small", width_mult: float = 1.0
):
    return build_baseline_video_enhancer(
        family="video_uformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ve(build_video_uformer_video_enhancer, "video_uformer_tiny")
