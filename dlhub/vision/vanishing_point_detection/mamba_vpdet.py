from __future__ import annotations

from torch import nn

from ._common import build_toy_vp_detector, smoke_test_vp_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_vpdet_tiny": {"width": 24, "depth": 1},
    "mamba_vpdet_small": {"width": 36, "depth": 2},
    "mamba_vpdet_base": {"width": 48, "depth": 3},
}


def build_mamba_vpdet_vp_detector(
    *, in_channels: int, variant: str = "mamba_vpdet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_vp_detector(
        family="mamba_vpdet",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vp_detector(build_mamba_vpdet_vp_detector, "mamba_vpdet_tiny")
