from __future__ import annotations
from torch import nn
from ._common import build_toy_image_to_video, smoke_test_image_to_video

_VARIANTS: dict[str, dict[str, int]] = {
    "motion_adapter_i2v_tiny": {"width": 24, "depth": 1, "frames": 4},
    "motion_adapter_i2v_small": {"width": 32, "depth": 2, "frames": 5},
    "motion_adapter_i2v_base": {"width": 48, "depth": 3, "frames": 6},
}

def build_motion_adapter_i2v_image_to_video(*, in_channels: int, variant: str = "motion_adapter_i2v_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_image_to_video(family="motion_adapter_i2v", mode="motion_adapter", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == "__main__":
    smoke_test_image_to_video(build_motion_adapter_i2v_image_to_video, "motion_adapter_i2v_tiny")
