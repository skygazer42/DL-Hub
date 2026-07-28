from __future__ import annotations
from ._common import build_baseline_mff, smoke_test_mff

_VARIANTS = {
    "mfnet_tiny": {"width": 24, "depth": 1},
    "mfnet_small": {"width": 32, "depth": 2},
    "mfnet_base": {"width": 48, "depth": 3},
}


def build_mfnet_multi_focus_fuser(
    *, in_channels: int, variant: str = "mfnet_small", width_mult: float = 1.0
):
    return build_baseline_mff(
        family="mfnet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_mff(build_mfnet_multi_focus_fuser, "mfnet_tiny")
