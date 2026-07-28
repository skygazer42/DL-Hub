from __future__ import annotations
from ._common import build_baseline_mff, smoke_test_mff

_VARIANTS = {
    "densemff_tiny": {"width": 24, "depth": 1},
    "densemff_small": {"width": 32, "depth": 2},
    "densemff_base": {"width": 48, "depth": 3},
}


def build_densemff_multi_focus_fuser(
    *, in_channels: int, variant: str = "densemff_small", width_mult: float = 1.0
):
    return build_baseline_mff(
        family="densemff",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_mff(build_densemff_multi_focus_fuser, "densemff_tiny")
