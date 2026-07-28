from __future__ import annotations
from torch import nn
from ._common import build_baseline_shadow_detector, smoke_test_shadow_detector

_VARIANTS: dict[str, dict[str, int]] = {
    "bdrar_shadow_tiny": {"width": 24, "depth": 1},
    "bdrar_shadow_small": {"width": 32, "depth": 2},
    "bdrar_shadow_base": {"width": 48, "depth": 3},
}


def build_bdrar_shadow_shadow_detector(
    *, in_channels: int, variant: str = "bdrar_shadow_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_shadow_detector(
        family="bdrar_shadow",
        mode="bdrar",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_shadow_detector(build_bdrar_shadow_shadow_detector, "bdrar_shadow_tiny")
