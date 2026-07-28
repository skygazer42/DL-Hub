from __future__ import annotations
from ._common import build_baseline_hw, smoke_test_hw

_VARIANTS = {
    "attention_hwr_tiny": {"width": 24, "depth": 1},
    "attention_hwr_small": {"width": 32, "depth": 2},
    "attention_hwr_base": {"width": 48, "depth": 3},
}


def build_attention_hwr_handwriting_recognizer(
    *,
    input_dim: int,
    vocab_size: int,
    variant: str = "attention_hwr_small",
    width_mult: float = 1.0,
):
    return build_baseline_hw(
        family="attention_hwr",
        variants=_VARIANTS,
        input_dim=int(input_dim),
        vocab_size=int(vocab_size),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hw(build_attention_hwr_handwriting_recognizer, "attention_hwr_tiny")
