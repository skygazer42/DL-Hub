from __future__ import annotations
from torch import nn
from ._common import build_toy_retriever, smoke_test_retriever

_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_retrieval_tiny": {"width": 24, "depth": 1},
    "transformer_retrieval_small": {"width": 32, "depth": 2},
    "transformer_retrieval_base": {"width": 48, "depth": 3},
}

def build_transformer_retrieval_retriever(*, in_channels: int, variant: str = "transformer_retrieval_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_retriever(family="transformer_retrieval", mode="transformer", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == "__main__":
    smoke_test_retriever(build_transformer_retrieval_retriever, "transformer_retrieval_tiny")
