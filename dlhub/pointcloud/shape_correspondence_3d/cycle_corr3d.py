from __future__ import annotations
from torch import nn
from ._common import build_toy_shape_correspondence_model, smoke_test_shape_correspondence_model

_VARIANTS: dict[str, dict[str, int]] = {
    "cycle_corr3d_tiny": {"width": 24, "depth": 1},
    "cycle_corr3d_small": {"width": 32, "depth": 2},
    "cycle_corr3d_base": {"width": 48, "depth": 3},
}

def build_cycle_corr3d_shape_correspondence_model(*, in_channels: int, variant: str = "cycle_corr3d_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_shape_correspondence_model(family="cycle_corr3d", mode="cycle", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == "__main__":
    smoke_test_shape_correspondence_model(build_cycle_corr3d_shape_correspondence_model, "cycle_corr3d_tiny")
