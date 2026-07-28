from __future__ import annotations
from ._common import build_baseline_hoi_detector, smoke_test_hoi

_VARIANTS = {
    "qpic_tiny": {"width": 24, "depth": 1},
    "qpic_small": {"width": 32, "depth": 2},
    "qpic_base": {"width": 48, "depth": 3},
}


def build_qpic_hoi_detector(
    *,
    in_channels: int,
    num_verbs: int,
    num_objects: int,
    variant: str = "qpic_small",
    width_mult: float = 1.0,
    num_queries: int = 16,
):
    return build_baseline_hoi_detector(
        family="qpic",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_verbs=int(num_verbs),
        num_objects=int(num_objects),
        variant=str(variant),
        width_mult=float(width_mult),
        num_queries=int(num_queries),
    )


if __name__ == "__main__":
    smoke_test_hoi(build_qpic_hoi_detector, "qpic_tiny")
