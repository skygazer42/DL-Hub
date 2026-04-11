from __future__ import annotations

from ._common import build_toy_reasoner, smoke_test_reasoner

_VARIANTS = {"transformer_reasoner_tiny": {"width": 24, "depth": 1, "steps": 2}, "transformer_reasoner_small": {"width": 32, "depth": 2, "steps": 3}, "transformer_reasoner_base": {"width": 48, "depth": 3, "steps": 4}}


def build_transformer_reasoner_reasoner(*, in_channels: int, variant: str = "transformer_reasoner_small", width_mult: float = 1.0):
    return build_toy_reasoner(family="transformer_reasoner", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_reasoner(build_transformer_reasoner_reasoner, "transformer_reasoner_tiny")
