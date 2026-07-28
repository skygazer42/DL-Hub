from __future__ import annotations

from torch import nn

from ._common import build_baseline_deweatherer, smoke_test_deweatherer


_VARIANTS: dict[str, dict[str, int]] = {
    "raindrop_removal_tiny": {"width": 24, "depth": 1, "passes": 2},
    "raindrop_removal_small": {"width": 32, "depth": 2, "passes": 2},
    "raindrop_removal_base": {"width": 48, "depth": 3, "passes": 3},
}


def build_raindrop_removal_deweatherer(
    *,
    in_channels: int,
    variant: str = "raindrop_removal_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_deweatherer(
        family="raindrop_removal",
        mode="raindrop",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_deweatherer(build_raindrop_removal_deweatherer, "raindrop_removal_tiny")
