from __future__ import annotations
from ._common import build_toy_video_enhancer, smoke_test_ve

_VARIANTS = {
    "fastdvdnet_tiny": {"width": 24, "depth": 1},
    "fastdvdnet_small": {"width": 32, "depth": 2},
    "fastdvdnet_base": {"width": 48, "depth": 3},
}


def build_fastdvdnet_video_enhancer(
    *, in_channels: int, variant: str = "fastdvdnet_small", width_mult: float = 1.0
):
    return build_toy_video_enhancer(
        family="fastdvdnet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ve(build_fastdvdnet_video_enhancer, "fastdvdnet_tiny")
