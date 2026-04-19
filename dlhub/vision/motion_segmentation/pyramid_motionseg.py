from __future__ import annotations

from torch import nn

from ._common import build_toy_motion_segmentor, smoke_test_motion_segmentor


_VARIANTS: dict[str, dict[str, int]] = {"pyramid_motionseg_tiny": {"width": 24, "depth": 1}, "pyramid_motionseg_small": {"width": 36, "depth": 2}, "pyramid_motionseg_base": {"width": 48, "depth": 3}}


def build_pyramid_motionseg_motion_segmentor(*, in_channels: int, variant: str = "pyramid_motionseg_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_motion_segmentor(family="pyramid_motionseg", mode="pyramid", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_motion_segmentor(build_pyramid_motionseg_motion_segmentor, "pyramid_motionseg_tiny")
