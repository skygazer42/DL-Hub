from __future__ import annotations
from ._common import build_toy_mff, smoke_test_mff

_VARIANTS = {
    "mambamff_tiny": {"width": 24, "depth": 1},
    "mambamff_small": {"width": 32, "depth": 2},
    "mambamff_base": {"width": 48, "depth": 3},
}


def build_mambamff_multi_focus_fuser(
    *, in_channels: int, variant: str = "mambamff_small", width_mult: float = 1.0
):
    return build_toy_mff(
        family="mambamff",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_mff(build_mambamff_multi_focus_fuser, "mambamff_tiny")
