from __future__ import annotations
from ._common import build_baseline_counter, smoke_test_counter

_VARIANTS = {
    "countr_pp_tiny": {"width": 24, "depth": 1},
    "countr_pp_small": {"width": 32, "depth": 2},
    "countr_pp_base": {"width": 48, "depth": 3},
}


def build_countr_pp_(
    *, in_channels: int, variant: str = "countr_pp_small", width_mult: float = 1.0
):
    return build_baseline_counter(
        family="countr_pp",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_counter(build_countr_pp_, "countr_pp_tiny")
