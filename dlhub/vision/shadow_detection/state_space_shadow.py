
from __future__ import annotations
from torch import nn
from ._common import build_toy_shadow_detector, smoke_test_shadow_detector

_VARIANTS: dict[str, dict[str, int]] = {
    "state_space_shadow_tiny": {"width": 24, "depth": 1},
    "state_space_shadow_small": {"width": 32, "depth": 2},
    "state_space_shadow_base": {"width": 48, "depth": 3},
}

def build_state_space_shadow_shadow_detector(*, in_channels: int, variant: str = "state_space_shadow_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_shadow_detector(family="state_space_shadow", mode="state_space", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == "__main__":
    smoke_test_shadow_detector(build_state_space_shadow_shadow_detector, "state_space_shadow_tiny")
