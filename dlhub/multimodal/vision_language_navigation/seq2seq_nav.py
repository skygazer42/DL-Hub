from __future__ import annotations
from torch import nn
from ._common import build_toy_navigator, smoke_test_navigator

_VARIANTS: dict[str, dict[str, int]] = {
    "seq2seq_nav_tiny": {"width": 24, "depth": 1},
    "seq2seq_nav_small": {"width": 32, "depth": 2},
    "seq2seq_nav_base": {"width": 48, "depth": 3},
}

def build_seq2seq_nav_navigator(*, in_channels: int, variant: str = "seq2seq_nav_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_navigator(family="seq2seq_nav", mode="seq2seq", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == "__main__":
    smoke_test_navigator(build_seq2seq_nav_navigator, "seq2seq_nav_tiny")
