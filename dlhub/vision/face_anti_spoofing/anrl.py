from __future__ import annotations
from ._common import build_baseline_spoofer, smoke_test_spoof

_VARIANTS = {
    "anrl_tiny": {"width": 24, "depth": 1},
    "anrl_small": {"width": 32, "depth": 2},
    "anrl_base": {"width": 48, "depth": 3},
}


def build_anrl_anti_spoofer(
    *, in_channels: int, variant: str = "anrl_small", width_mult: float = 1.0
):
    return build_baseline_spoofer(
        family="anrl",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_spoof(build_anrl_anti_spoofer, "anrl_tiny")
