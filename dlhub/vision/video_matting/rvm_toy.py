from __future__ import annotations

from ._common import build_toy_video_matter, smoke_test_video_matter

_VARIANTS = {
    "rvm_toy_tiny": {"width": 24, "depth": 1},
    "rvm_toy_small": {"width": 32, "depth": 2},
    "rvm_toy_base": {"width": 48, "depth": 3},
}


def build_rvm_toy_video_matter(
    *, in_channels: int, variant: str = "rvm_toy_small", width_mult: float = 1.0
):
    return build_toy_video_matter(
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_video_matter(build_rvm_toy_video_matter, "rvm_toy_tiny")
