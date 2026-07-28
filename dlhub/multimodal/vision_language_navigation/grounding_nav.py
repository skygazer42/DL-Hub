from __future__ import annotations
from torch import nn
from ._common import build_baseline_navigator, smoke_test_navigator

_VARIANTS: dict[str, dict[str, int]] = {
    "grounding_nav_tiny": {"width": 24, "depth": 1},
    "grounding_nav_small": {"width": 32, "depth": 2},
    "grounding_nav_base": {"width": 48, "depth": 3},
}


def build_grounding_nav_navigator(
    *, in_channels: int, variant: str = "grounding_nav_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_navigator(
        family="grounding_nav",
        mode="grounding",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_navigator(build_grounding_nav_navigator, "grounding_nav_tiny")
