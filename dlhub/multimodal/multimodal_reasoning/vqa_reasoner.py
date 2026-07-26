from __future__ import annotations

from ._common import build_toy_reasoner, smoke_test_reasoner

_VARIANTS = {
    "vqa_reasoner_tiny": {"width": 24, "depth": 1, "steps": 2},
    "vqa_reasoner_small": {"width": 32, "depth": 2, "steps": 3},
    "vqa_reasoner_base": {"width": 48, "depth": 3, "steps": 4},
}


def build_vqa_reasoner_reasoner(
    *, in_channels: int, variant: str = "vqa_reasoner_small", width_mult: float = 1.0
):
    return build_toy_reasoner(
        family="vqa_reasoner",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_reasoner(build_vqa_reasoner_reasoner, "vqa_reasoner_tiny")
