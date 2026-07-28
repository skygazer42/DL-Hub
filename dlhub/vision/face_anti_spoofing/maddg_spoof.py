from __future__ import annotations
from ._common import build_baseline_spoofer, smoke_test_spoof

_VARIANTS = {
    "maddg_spoof_tiny": {"width": 24, "depth": 1},
    "maddg_spoof_small": {"width": 32, "depth": 2},
    "maddg_spoof_base": {"width": 48, "depth": 3},
}


def build_maddg_spoof_anti_spoofer(
    *, in_channels: int, variant: str = "maddg_spoof_small", width_mult: float = 1.0
):
    return build_baseline_spoofer(
        family="maddg_spoof",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_spoof(build_maddg_spoof_anti_spoofer, "maddg_spoof_tiny")
