from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "page_curve_net_tiny": {"width": 24, "depth": 1},
    "page_curve_net_small": {"width": 32, "depth": 2},
    "page_curve_net_base": {"width": 48, "depth": 3},
}


def build_page_curve_net_dewarper(
    *, in_channels: int, variant: str = "page_curve_net_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="page_curve_net",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_page_curve_net_dewarper, "page_curve_net_tiny")
