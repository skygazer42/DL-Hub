from __future__ import annotations
from torch import nn
from ._common import build_baseline_restoration, smoke_test_restoration

_VARIANTS = {
    "stackdmphn_tiny": {"width": 24, "depth": 1},
    "stackdmphn_small": {"width": 32, "depth": 2},
    "stackdmphn_base": {"width": 48, "depth": 3},
}


def build_stackdmphn_deblurrer(
    *, in_channels: int, variant: str = "stackdmphn_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_restoration(
        family="stackdmphn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        out_key="deblurred",
    )


if __name__ == "__main__":
    smoke_test_restoration(build_stackdmphn_deblurrer, "stackdmphn_tiny", "deblurred")
