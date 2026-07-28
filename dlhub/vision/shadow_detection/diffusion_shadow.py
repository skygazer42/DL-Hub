from __future__ import annotations
from torch import nn
from ._common import build_baseline_shadow_detector, smoke_test_shadow_detector

_VARIANTS: dict[str, dict[str, int]] = {
    "diffusion_shadow_tiny": {"width": 24, "depth": 1},
    "diffusion_shadow_small": {"width": 32, "depth": 2},
    "diffusion_shadow_base": {"width": 48, "depth": 3},
}


def build_diffusion_shadow_shadow_detector(
    *, in_channels: int, variant: str = "diffusion_shadow_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_shadow_detector(
        family="diffusion_shadow",
        mode="diffusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_shadow_detector(build_diffusion_shadow_shadow_detector, "diffusion_shadow_tiny")
