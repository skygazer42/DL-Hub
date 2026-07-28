from __future__ import annotations

from ._common import build_baseline_reasoner, smoke_test_reasoner

_VARIANTS = {
    "tool_reasoner_tiny": {"width": 24, "depth": 1, "steps": 2},
    "tool_reasoner_small": {"width": 32, "depth": 2, "steps": 3},
    "tool_reasoner_base": {"width": 48, "depth": 3, "steps": 4},
}


def build_tool_reasoner_reasoner(
    *, in_channels: int, variant: str = "tool_reasoner_small", width_mult: float = 1.0
):
    return build_baseline_reasoner(
        family="tool_reasoner",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_reasoner(build_tool_reasoner_reasoner, "tool_reasoner_tiny")
