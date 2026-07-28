from __future__ import annotations

from torch import nn

from ._common import build_baseline_mirror_segmentor, smoke_test_mirror_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "pyramid_mirrorseg_tiny": {"width": 24, "depth": 1},
    "pyramid_mirrorseg_small": {"width": 36, "depth": 2},
    "pyramid_mirrorseg_base": {"width": 48, "depth": 3},
}


def build_pyramid_mirrorseg_mirror_segmentor(
    *, in_channels: int, variant: str = "pyramid_mirrorseg_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_mirror_segmentor(
        family="pyramid_mirrorseg",
        mode="pyramid",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_mirror_segmentor(build_pyramid_mirrorseg_mirror_segmentor, "pyramid_mirrorseg_tiny")
