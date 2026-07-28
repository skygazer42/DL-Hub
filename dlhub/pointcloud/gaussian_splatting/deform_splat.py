from __future__ import annotations

from ._common import build_baseline_splatter, smoke_test_splatter

_VARIANTS = {
    "deform_splat_tiny": {"width": 24, "depth": 1},
    "deform_splat_small": {"width": 32, "depth": 2},
    "deform_splat_base": {"width": 48, "depth": 3},
}


def build_deform_splat_splatter(
    *, in_channels: int, variant: str = "deform_splat_small", width_mult: float = 1.0
):
    return build_baseline_splatter(
        family="deform_splat",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_splatter(build_deform_splat_splatter, "deform_splat_tiny")
