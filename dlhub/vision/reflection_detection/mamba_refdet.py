from __future__ import annotations

from torch import nn

from ._common import build_toy_reflection_detector, smoke_test_reflection_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_refdet_tiny": {"width": 24, "depth": 1},
    "mamba_refdet_small": {"width": 36, "depth": 2},
    "mamba_refdet_base": {"width": 48, "depth": 3},
}


def build_mamba_refdet_reflection_detector(
    *, in_channels: int, variant: str = "mamba_refdet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_reflection_detector(
        family="mamba_refdet",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_reflection_detector(build_mamba_refdet_reflection_detector, "mamba_refdet_tiny")
