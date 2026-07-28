from __future__ import annotations
from torch import nn
from ._common import build_baseline_retriever, smoke_test_retriever

_VARIANTS: dict[str, dict[str, int]] = {
    "clip_retrieval_tiny": {"width": 24, "depth": 1},
    "clip_retrieval_small": {"width": 32, "depth": 2},
    "clip_retrieval_base": {"width": 48, "depth": 3},
}


def build_clip_retrieval_retriever(
    *, in_channels: int, variant: str = "clip_retrieval_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_retriever(
        family="clip_retrieval",
        mode="clip",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_retriever(build_clip_retrieval_retriever, "clip_retrieval_tiny")
