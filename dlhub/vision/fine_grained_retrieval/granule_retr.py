from __future__ import annotations
from torch import nn
from ._common import build_compact_retrieval_model, smoke_test_retrieval

_VARIANTS = {
    "granule_retr_tiny": {"width": 24, "depth": 1, "embed": 128},
    "granule_retr_small": {"width": 32, "depth": 2, "embed": 160},
    "granule_retr_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_granule_retr_(
    *, in_channels: int, variant: str = "granule_retr_small", width_mult: float = 1.0
) -> nn.Module:
    return build_compact_retrieval_model(
        family="granule_retr",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_retrieval(build_granule_retr_, "granule_retr_tiny")
