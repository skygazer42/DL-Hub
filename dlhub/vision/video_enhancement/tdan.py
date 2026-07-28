from __future__ import annotations
from ._common import build_baseline_video_enhancer, smoke_test_ve

_VARIANTS = {
    "tdan_tiny": {"width": 24, "depth": 1},
    "tdan_small": {"width": 32, "depth": 2},
    "tdan_base": {"width": 48, "depth": 3},
}


def build_tdan_video_enhancer(
    *, in_channels: int, variant: str = "tdan_small", width_mult: float = 1.0
):
    return build_baseline_video_enhancer(
        family="tdan",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ve(build_tdan_video_enhancer, "tdan_tiny")
