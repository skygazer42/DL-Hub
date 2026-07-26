from __future__ import annotations
from torch import nn
from ._common import build_toy_navigator, smoke_test_navigator

_VARIANTS: dict[str, dict[str, int]] = {
    "object_goal_nav_tiny": {"width": 24, "depth": 1},
    "object_goal_nav_small": {"width": 32, "depth": 2},
    "object_goal_nav_base": {"width": 48, "depth": 3},
}


def build_object_goal_nav_navigator(
    *, in_channels: int, variant: str = "object_goal_nav_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_navigator(
        family="object_goal_nav",
        mode="object_goal",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_navigator(build_object_goal_nav_navigator, "object_goal_nav_tiny")
