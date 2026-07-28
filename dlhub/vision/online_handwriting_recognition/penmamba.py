from __future__ import annotations
from ._common import build_baseline_hw, smoke_test_hw

_VARIANTS = {
    "penmamba_tiny": {"width": 24, "depth": 1},
    "penmamba_small": {"width": 32, "depth": 2},
    "penmamba_base": {"width": 48, "depth": 3},
}


def build_penmamba_handwriting_recognizer(
    *, input_dim: int, vocab_size: int, variant: str = "penmamba_small", width_mult: float = 1.0
):
    return build_baseline_hw(
        family="penmamba",
        variants=_VARIANTS,
        input_dim=int(input_dim),
        vocab_size=int(vocab_size),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hw(build_penmamba_handwriting_recognizer, "penmamba_tiny")
