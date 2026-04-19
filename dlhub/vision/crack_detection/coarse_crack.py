from __future__ import annotations

from torch import nn

from ._common import build_toy_crack_detector, smoke_test_crack_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "coarse_crack_tiny": {"width": 24, "depth": 1, "classes": 2},
    "coarse_crack_small": {"width": 36, "depth": 2, "classes": 2},
    "coarse_crack_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_coarse_crack_crack_detector(
    *, in_channels: int, variant: str = "coarse_crack_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_crack_detector(
        family="coarse_crack",
        mode="coarse",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_crack_detector(build_coarse_crack_crack_detector, "coarse_crack_tiny")
