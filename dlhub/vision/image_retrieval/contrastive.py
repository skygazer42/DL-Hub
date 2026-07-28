from __future__ import annotations
from torch import nn
from ._common import build_baseline_retrieval_model, smoke_test_retrieval

_VARIANTS = {
    "contrastive_tiny": {"width": 24, "depth": 1, "embed": 128},
    "contrastive_small": {"width": 32, "depth": 2, "embed": 160},
    "contrastive_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_contrastive_retrieval_model(
    *, in_channels: int, variant: str = "contrastive_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_retrieval_model(
        family="contrastive",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_retrieval(build_contrastive_retrieval_model, "contrastive_tiny")
