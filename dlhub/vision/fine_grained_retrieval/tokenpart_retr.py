from __future__ import annotations
from torch import nn
from ._common import build_baseline_retrieval_model, smoke_test_retrieval

_VARIANTS = {
    "tokenpart_retr_tiny": {"width": 24, "depth": 1, "embed": 128},
    "tokenpart_retr_small": {"width": 32, "depth": 2, "embed": 160},
    "tokenpart_retr_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_tokenpart_retr_(
    *, in_channels: int, variant: str = "tokenpart_retr_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_retrieval_model(
        family="tokenpart_retr",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_retrieval(build_tokenpart_retr_, "tokenpart_retr_tiny")
