from __future__ import annotations
from ._common import build_toy_editor, smoke_test_editor

_VARIANTS = {
    "brushnet_edit_tiny": {"width": 24, "depth": 1},
    "brushnet_edit_small": {"width": 32, "depth": 2},
    "brushnet_edit_base": {"width": 48, "depth": 3},
}


def build_brushnet_edit_editor(
    *, in_channels: int, variant: str = "brushnet_edit_small", width_mult: float = 1.0
):
    return build_toy_editor(
        family="brushnet_edit",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_editor(build_brushnet_edit_editor, "brushnet_edit_tiny")
