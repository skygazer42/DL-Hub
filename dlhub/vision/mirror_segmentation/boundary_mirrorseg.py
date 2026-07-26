from __future__ import annotations

from torch import nn

from ._common import build_toy_mirror_segmentor, smoke_test_mirror_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "boundary_mirrorseg_tiny": {"width": 24, "depth": 1},
    "boundary_mirrorseg_small": {"width": 36, "depth": 2},
    "boundary_mirrorseg_base": {"width": 48, "depth": 3},
}


def build_boundary_mirrorseg_mirror_segmentor(
    *, in_channels: int, variant: str = "boundary_mirrorseg_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_mirror_segmentor(
        family="boundary_mirrorseg",
        mode="boundary",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_mirror_segmentor(
        build_boundary_mirrorseg_mirror_segmentor, "boundary_mirrorseg_tiny"
    )
