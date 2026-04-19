from __future__ import annotations

from torch import nn

from ._common import build_toy_crack_detector, smoke_test_crack_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "unet_crack_tiny": {"width": 24, "depth": 1, "classes": 2},
    "unet_crack_small": {"width": 36, "depth": 2, "classes": 2},
    "unet_crack_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_unet_crack_crack_detector(
    *, in_channels: int, variant: str = "unet_crack_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_crack_detector(
        family="unet_crack",
        mode="unet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_crack_detector(build_unet_crack_crack_detector, "unet_crack_tiny")
