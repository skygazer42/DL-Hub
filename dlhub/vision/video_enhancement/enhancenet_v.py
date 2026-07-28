from __future__ import annotations
from ._common import build_baseline_video_enhancer, smoke_test_ve

_VARIANTS = {
    "enhancenet_v_tiny": {"width": 24, "depth": 1},
    "enhancenet_v_small": {"width": 32, "depth": 2},
    "enhancenet_v_base": {"width": 48, "depth": 3},
}


def build_enhancenet_v_video_enhancer(
    *, in_channels: int, variant: str = "enhancenet_v_small", width_mult: float = 1.0
):
    return build_baseline_video_enhancer(
        family="enhancenet_v",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ve(build_enhancenet_v_video_enhancer, "enhancenet_v_tiny")
