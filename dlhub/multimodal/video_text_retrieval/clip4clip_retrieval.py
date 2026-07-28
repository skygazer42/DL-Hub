from __future__ import annotations

from torch import nn

from ._common import build_baseline_vtr, smoke_test_vtr

_VARIANTS: dict[str, dict[str, int]] = {
    "clip4clip_retrieval_tiny": {"width": 24, "depth": 1},
    "clip4clip_retrieval_small": {"width": 32, "depth": 2},
    "clip4clip_retrieval_base": {"width": 48, "depth": 3},
}


def build_clip4clip_retrieval_retriever(
    *, in_channels: int, variant: str = "clip4clip_retrieval_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_vtr(
        family="clip4clip_retrieval",
        mode="clip4clip",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vtr(build_clip4clip_retrieval_retriever, "clip4clip_retrieval_tiny")
