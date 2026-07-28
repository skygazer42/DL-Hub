from __future__ import annotations
from torch import nn
from ._common import build_baseline_image_to_video, smoke_test_image_to_video

_VARIANTS: dict[str, dict[str, int]] = {
    "dynami_baseline_tiny": {"width": 24, "depth": 1, "frames": 4},
    "dynami_baseline_small": {"width": 32, "depth": 2, "frames": 5},
    "dynami_baseline_base": {"width": 48, "depth": 3, "frames": 6},
}


def build_dynami_baseline_image_to_video(
    *, in_channels: int, variant: str = "dynami_baseline_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_image_to_video(
        family="dynami_baseline",
        mode="dynami",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_image_to_video(build_dynami_baseline_image_to_video, "dynami_baseline_tiny")
