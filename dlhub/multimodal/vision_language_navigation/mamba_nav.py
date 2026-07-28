from __future__ import annotations
from torch import nn
from ._common import build_baseline_navigator, smoke_test_navigator

_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_nav_tiny": {"width": 24, "depth": 1},
    "mamba_nav_small": {"width": 32, "depth": 2},
    "mamba_nav_base": {"width": 48, "depth": 3},
}


def build_mamba_nav_navigator(
    *, in_channels: int, variant: str = "mamba_nav_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_navigator(
        family="mamba_nav",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_navigator(build_mamba_nav_navigator, "mamba_nav_tiny")
