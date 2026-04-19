from __future__ import annotations

from torch import nn

from ._common import build_toy_reflection_detector, smoke_test_reflection_detector


_VARIANTS: dict[str, dict[str, int]] = {"water_refdet_tiny": {"width": 24, "depth": 1}, "water_refdet_small": {"width": 36, "depth": 2}, "water_refdet_base": {"width": 48, "depth": 3}}


def build_water_refdet_reflection_detector(*, in_channels: int, variant: str = "water_refdet_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_reflection_detector(family="water_refdet", mode="water", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_reflection_detector(build_water_refdet_reflection_detector, "water_refdet_tiny")
