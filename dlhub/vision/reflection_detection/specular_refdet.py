from __future__ import annotations

from torch import nn

from ._common import build_baseline_reflection_detector, smoke_test_reflection_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "specular_refdet_tiny": {"width": 24, "depth": 1},
    "specular_refdet_small": {"width": 36, "depth": 2},
    "specular_refdet_base": {"width": 48, "depth": 3},
}


def build_specular_refdet_reflection_detector(
    *, in_channels: int, variant: str = "specular_refdet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_reflection_detector(
        family="specular_refdet",
        mode="specular",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_reflection_detector(
        build_specular_refdet_reflection_detector, "specular_refdet_tiny"
    )
