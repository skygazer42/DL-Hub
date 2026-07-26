from __future__ import annotations

from ._common import build_toy_reasoner, smoke_test_reasoner

_VARIANTS = {
    "program_reasoner_tiny": {"width": 24, "depth": 1, "steps": 3},
    "program_reasoner_small": {"width": 32, "depth": 2, "steps": 4},
    "program_reasoner_base": {"width": 48, "depth": 3, "steps": 5},
}


def build_program_reasoner_reasoner(
    *, in_channels: int, variant: str = "program_reasoner_small", width_mult: float = 1.0
):
    return build_toy_reasoner(
        family="program_reasoner",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_reasoner(build_program_reasoner_reasoner, "program_reasoner_tiny")
