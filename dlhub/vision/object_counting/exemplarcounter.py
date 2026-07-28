from __future__ import annotations
from ._common import build_baseline_counter, smoke_test_counter

_VARIANTS = {
    "exemplarcounter_tiny": {"width": 24, "depth": 1},
    "exemplarcounter_small": {"width": 32, "depth": 2},
    "exemplarcounter_base": {"width": 48, "depth": 3},
}


def build_exemplarcounter_(
    *, in_channels: int, variant: str = "exemplarcounter_small", width_mult: float = 1.0
):
    return build_baseline_counter(
        family="exemplarcounter",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_counter(build_exemplarcounter_, "exemplarcounter_tiny")
