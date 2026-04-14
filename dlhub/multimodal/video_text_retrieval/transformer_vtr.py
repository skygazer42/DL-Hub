from __future__ import annotations

from torch import nn

from ._common import build_toy_vtr, smoke_test_vtr

_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_vtr_tiny": {"width": 24, "depth": 1},
    "transformer_vtr_small": {"width": 32, "depth": 2},
    "transformer_vtr_base": {"width": 48, "depth": 3},
}


def build_transformer_vtr_retriever(
    *, in_channels: int, variant: str = "transformer_vtr_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_vtr(
        family="transformer_vtr",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vtr(build_transformer_vtr_retriever, "transformer_vtr_tiny")
