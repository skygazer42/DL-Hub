from __future__ import annotations
from ._common import build_baseline_inter, smoke_test_inter

_VARIANTS = {
    "seqprompt_tiny": {"width": 24, "depth": 1},
    "seqprompt_small": {"width": 32, "depth": 2},
    "seqprompt_base": {"width": 48, "depth": 3},
}


def build_seqprompt_(
    *, in_channels: int, variant: str = "seqprompt_small", width_mult: float = 1.0
):
    return build_baseline_inter(
        family="seqprompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_inter(build_seqprompt_, "seqprompt_tiny")
