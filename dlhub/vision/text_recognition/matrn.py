from __future__ import annotations
from ._common import build_baseline_text_recognizer, smoke_test_rec

_VARIANTS = {
    "matrn_tiny": {"width": 32, "depth": 1},
    "matrn_small": {"width": 48, "depth": 2},
    "matrn_base": {"width": 64, "depth": 3},
}


def build_matrn_text_recognizer(
    *,
    in_channels: int,
    vocab_size: int,
    seq_len: int = 16,
    variant: str = "matrn_small",
    width_mult: float = 1.0,
):
    return build_baseline_text_recognizer(
        family="matrn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        variant=str(variant),
        width_mult=float(width_mult),
        decoder="transformer",
    )


if __name__ == "__main__":
    smoke_test_rec(build_matrn_text_recognizer, "matrn_tiny")
