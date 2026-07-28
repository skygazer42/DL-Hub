from __future__ import annotations
from torch import nn
from ._common import build_baseline_retriever, smoke_test_retriever

_VARIANTS: dict[str, dict[str, int]] = {
    "diffusion_retrieval_tiny": {"width": 24, "depth": 1},
    "diffusion_retrieval_small": {"width": 32, "depth": 2},
    "diffusion_retrieval_base": {"width": 48, "depth": 3},
}


def build_diffusion_retrieval_retriever(
    *, in_channels: int, variant: str = "diffusion_retrieval_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_retriever(
        family="diffusion_retrieval",
        mode="diffusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_retriever(build_diffusion_retrieval_retriever, "diffusion_retrieval_tiny")
