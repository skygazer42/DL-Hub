from __future__ import annotations
from torch import nn
from ._common import build_toy_retrieval_model, smoke_test_retrieval

_VARIANTS = {
    "apgem_vpr_tiny": {"width": 24, "depth": 1, "embed": 128},
    "apgem_vpr_small": {"width": 32, "depth": 2, "embed": 160},
    "apgem_vpr_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_apgem_vpr_(
    *, in_channels: int, variant: str = "apgem_vpr_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_retrieval_model(
        family="apgem_vpr",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_retrieval(build_apgem_vpr_, "apgem_vpr_tiny")
