from __future__ import annotations
from ._common import build_baseline_text_recognizer, smoke_test_rec

_VARIANTS = {
    "cppd_tiny": {"width": 32, "depth": 1},
    "cppd_small": {"width": 48, "depth": 2},
    "cppd_base": {"width": 64, "depth": 3},
}


def build_cppd_text_recognizer(
    *,
    in_channels: int,
    vocab_size: int,
    seq_len: int = 16,
    variant: str = "cppd_small",
    width_mult: float = 1.0,
):
    return build_baseline_text_recognizer(
        family="cppd",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        variant=str(variant),
        width_mult=float(width_mult),
        decoder="gru",
    )


if __name__ == "__main__":
    smoke_test_rec(build_cppd_text_recognizer, "cppd_tiny")
