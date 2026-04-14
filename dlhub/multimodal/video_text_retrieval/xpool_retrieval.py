from __future__ import annotations

from torch import nn

from ._common import build_toy_vtr, smoke_test_vtr

_VARIANTS: dict[str, dict[str, int]] = {
    "xpool_retrieval_tiny": {"width": 24, "depth": 1},
    "xpool_retrieval_small": {"width": 32, "depth": 2},
    "xpool_retrieval_base": {"width": 48, "depth": 3},
}


def build_xpool_retrieval_retriever(
    *, in_channels: int, variant: str = "xpool_retrieval_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_vtr(
        family="xpool_retrieval",
        mode="xpool",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vtr(build_xpool_retrieval_retriever, "xpool_retrieval_tiny")
