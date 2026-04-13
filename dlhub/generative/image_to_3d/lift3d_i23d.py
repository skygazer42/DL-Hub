from __future__ import annotations
from torch import nn
from ._common import build_toy_image_to_3d_generator, smoke_test_image_to_3d

_VARIANTS: dict[str, dict[str, int]] = {
    "lift3d_i23d_tiny": {"width": 24, "depth": 1, "voxel_size": 8},
    "lift3d_i23d_small": {"width": 32, "depth": 2, "voxel_size": 10},
    "lift3d_i23d_base": {"width": 48, "depth": 3, "voxel_size": 12},
}

def build_lift3d_i23d_image_to_3d_generator(*, in_channels: int, variant: str = "lift3d_i23d_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_image_to_3d_generator(family="lift3d_i23d", mode="lift3d", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == "__main__":
    smoke_test_image_to_3d(build_lift3d_i23d_image_to_3d_generator, "lift3d_i23d_tiny")
