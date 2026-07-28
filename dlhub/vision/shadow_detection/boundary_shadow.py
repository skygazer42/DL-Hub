from __future__ import annotations
from torch import nn
from ._common import build_baseline_shadow_detector, smoke_test_shadow_detector

_VARIANTS: dict[str, dict[str, int]] = {
    "boundary_shadow_tiny": {"width": 24, "depth": 1},
    "boundary_shadow_small": {"width": 32, "depth": 2},
    "boundary_shadow_base": {"width": 48, "depth": 3},
}


def build_boundary_shadow_shadow_detector(
    *, in_channels: int, variant: str = "boundary_shadow_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_shadow_detector(
        family="boundary_shadow",
        mode="boundary",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_shadow_detector(build_boundary_shadow_shadow_detector, "boundary_shadow_tiny")
