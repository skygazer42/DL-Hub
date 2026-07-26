from __future__ import annotations
from torch import nn
from ._common import build_toy_restoration, smoke_test_restoration

_VARIANTS = {
    "f3net_tiny": {"width": 24, "depth": 1},
    "f3net_small": {"width": 32, "depth": 2},
    "f3net_base": {"width": 48, "depth": 3},
}


def build_f3net_saliency_detector(
    *, in_channels: int, variant: str = "f3net_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_restoration(
        family="f3net",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        out_key="saliency",
    )


if __name__ == "__main__":
    smoke_test_restoration(build_f3net_saliency_detector, "f3net_tiny", "saliency")
