from __future__ import annotations
from torch import nn
from ._common import build_baseline_restoration, smoke_test_restoration

_VARIANTS = {
    "basnet_tiny": {"width": 24, "depth": 1},
    "basnet_small": {"width": 32, "depth": 2},
    "basnet_base": {"width": 48, "depth": 3},
}


def build_basnet_saliency_detector(
    *, in_channels: int, variant: str = "basnet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_restoration(
        family="basnet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        out_key="saliency",
    )


if __name__ == "__main__":
    smoke_test_restoration(build_basnet_saliency_detector, "basnet_tiny", "saliency")
