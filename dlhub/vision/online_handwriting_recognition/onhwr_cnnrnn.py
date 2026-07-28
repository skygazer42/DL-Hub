from __future__ import annotations
from ._common import build_baseline_hw, smoke_test_hw

_VARIANTS = {
    "onhwr_cnnrnn_tiny": {"width": 24, "depth": 1},
    "onhwr_cnnrnn_small": {"width": 32, "depth": 2},
    "onhwr_cnnrnn_base": {"width": 48, "depth": 3},
}


def build_onhwr_cnnrnn_handwriting_recognizer(
    *, input_dim: int, vocab_size: int, variant: str = "onhwr_cnnrnn_small", width_mult: float = 1.0
):
    return build_baseline_hw(
        family="onhwr_cnnrnn",
        variants=_VARIANTS,
        input_dim=int(input_dim),
        vocab_size=int(vocab_size),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hw(build_onhwr_cnnrnn_handwriting_recognizer, "onhwr_cnnrnn_tiny")
