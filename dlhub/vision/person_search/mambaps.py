from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "mambaps_tiny": {"width": 24, "depth": 1},
    "mambaps_small": {"width": 32, "depth": 2},
    "mambaps_base": {"width": 48, "depth": 3},
}


def build_mambaps_person_searcher(
    *, in_channels: int, variant: str = "mambaps_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="mambaps",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_mambaps_person_searcher, "mambaps_tiny")
