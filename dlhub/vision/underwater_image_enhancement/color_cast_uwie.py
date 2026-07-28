from __future__ import annotations

from torch import nn

from ._common import build_baseline_enhancer, smoke_test_enhancer


_VARIANTS: dict[str, dict[str, int]] = {
    "color_cast_uwie_tiny": {"width": 24, "depth": 1, "steps": 1},
    "color_cast_uwie_small": {"width": 36, "depth": 2, "steps": 1},
    "color_cast_uwie_base": {"width": 48, "depth": 3, "steps": 2},
}


def build_color_cast_uwie_enhancer(
    *,
    in_channels: int,
    variant: str = "color_cast_uwie_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_enhancer(
        family="color_cast_uwie",
        mode="color_cast",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_enhancer(build_color_cast_uwie_enhancer, "color_cast_uwie_tiny")
