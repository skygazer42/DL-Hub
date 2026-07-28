from __future__ import annotations
from ._common import build_baseline_plate, smoke_test_plate

_VARIANTS = {
    "mambalpr_tiny": {"width": 24, "depth": 1},
    "mambalpr_small": {"width": 32, "depth": 2},
    "mambalpr_base": {"width": 48, "depth": 3},
}


def build_mambalpr_plate_recognizer(
    *,
    in_channels: int,
    vocab_size: int,
    seq_len: int = 10,
    variant: str = "mambalpr_small",
    width_mult: float = 1.0,
):
    return build_baseline_plate(
        family="mambalpr",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_plate(build_mambalpr_plate_recognizer, "mambalpr_tiny")
