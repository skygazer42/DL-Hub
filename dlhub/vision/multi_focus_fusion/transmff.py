from __future__ import annotations
from ._common import build_baseline_mff, smoke_test_mff

_VARIANTS = {
    "transmff_tiny": {"width": 24, "depth": 1},
    "transmff_small": {"width": 32, "depth": 2},
    "transmff_base": {"width": 48, "depth": 3},
}


def build_transmff_multi_focus_fuser(
    *, in_channels: int, variant: str = "transmff_small", width_mult: float = 1.0
):
    return build_baseline_mff(
        family="transmff",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_mff(build_transmff_multi_focus_fuser, "transmff_tiny")
