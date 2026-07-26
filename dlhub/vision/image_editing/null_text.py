from __future__ import annotations
from ._common import build_toy_editor, smoke_test_editor

_VARIANTS = {
    "null_text_tiny": {"width": 24, "depth": 1},
    "null_text_small": {"width": 32, "depth": 2},
    "null_text_base": {"width": 48, "depth": 3},
}


def build_null_text_editor(
    *, in_channels: int, variant: str = "null_text_small", width_mult: float = 1.0
):
    return build_toy_editor(
        family="null_text",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_editor(build_null_text_editor, "null_text_tiny")
