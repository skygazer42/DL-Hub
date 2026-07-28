from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "dewarp_net_tiny": {"width": 24, "depth": 1},
    "dewarp_net_small": {"width": 32, "depth": 2},
    "dewarp_net_base": {"width": 48, "depth": 3},
}


def build_dewarp_net_dewarper(
    *, in_channels: int, variant: str = "dewarp_net_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="dewarp_net",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_dewarp_net_dewarper, "dewarp_net_tiny")
