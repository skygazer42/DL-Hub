from __future__ import annotations
from torch import nn
from ._common import build_baseline_navigator, smoke_test_navigator

_VARIANTS: dict[str, dict[str, int]] = {
    "map_memory_nav_tiny": {"width": 24, "depth": 1},
    "map_memory_nav_small": {"width": 32, "depth": 2},
    "map_memory_nav_base": {"width": 48, "depth": 3},
}


def build_map_memory_nav_navigator(
    *, in_channels: int, variant: str = "map_memory_nav_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_navigator(
        family="map_memory_nav",
        mode="map_memory",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_navigator(build_map_memory_nav_navigator, "map_memory_nav_tiny")
