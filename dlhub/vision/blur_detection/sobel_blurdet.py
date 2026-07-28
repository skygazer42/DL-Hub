from __future__ import annotations

from torch import nn

from ._common import build_baseline_blur_detector, smoke_test_blur_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "sobel_blurdet_tiny": {"width": 24, "depth": 1, "bins": 6},
    "sobel_blurdet_small": {"width": 36, "depth": 2, "bins": 8},
    "sobel_blurdet_base": {"width": 48, "depth": 3, "bins": 10},
}


def build_sobel_blurdet_blur_detector(
    *, in_channels: int, variant: str = "sobel_blurdet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_blur_detector(
        family="sobel_blurdet",
        mode="sobel",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_blur_detector(build_sobel_blurdet_blur_detector, "sobel_blurdet_tiny")
