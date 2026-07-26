from __future__ import annotations

from torch import nn

from ._common import build_toy_blur_detector, smoke_test_blur_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "laplacian_blurdet_tiny": {"width": 24, "depth": 1, "bins": 6},
    "laplacian_blurdet_small": {"width": 36, "depth": 2, "bins": 8},
    "laplacian_blurdet_base": {"width": 48, "depth": 3, "bins": 10},
}


def build_laplacian_blurdet_blur_detector(
    *, in_channels: int, variant: str = "laplacian_blurdet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_blur_detector(
        family="laplacian_blurdet",
        mode="laplacian",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_blur_detector(build_laplacian_blurdet_blur_detector, "laplacian_blurdet_tiny")
