from __future__ import annotations
from ._common import build_toy_mff, smoke_test_mff

_VARIANTS = {
    "swinmff_tiny": {"width": 24, "depth": 1},
    "swinmff_small": {"width": 32, "depth": 2},
    "swinmff_base": {"width": 48, "depth": 3},
}


def build_swinmff_multi_focus_fuser(
    *, in_channels: int, variant: str = "swinmff_small", width_mult: float = 1.0
):
    return build_toy_mff(
        family="swinmff",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_mff(build_swinmff_multi_focus_fuser, "swinmff_tiny")
