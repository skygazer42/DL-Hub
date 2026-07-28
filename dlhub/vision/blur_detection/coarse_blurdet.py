from __future__ import annotations

from torch import nn

from ._common import build_baseline_blur_detector, smoke_test_blur_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "coarse_blurdet_tiny": {"width": 24, "depth": 1, "bins": 6},
    "coarse_blurdet_small": {"width": 36, "depth": 2, "bins": 8},
    "coarse_blurdet_base": {"width": 48, "depth": 3, "bins": 10},
}


def build_coarse_blurdet_blur_detector(
    *, in_channels: int, variant: str = "coarse_blurdet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_blur_detector(
        family="coarse_blurdet",
        mode="coarse",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_blur_detector(build_coarse_blurdet_blur_detector, "coarse_blurdet_tiny")
