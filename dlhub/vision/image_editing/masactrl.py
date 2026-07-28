from __future__ import annotations
from ._common import build_baseline_editor, smoke_test_editor

_VARIANTS = {
    "masactrl_tiny": {"width": 24, "depth": 1},
    "masactrl_small": {"width": 32, "depth": 2},
    "masactrl_base": {"width": 48, "depth": 3},
}


def build_masactrl_editor(
    *, in_channels: int, variant: str = "masactrl_small", width_mult: float = 1.0
):
    return build_baseline_editor(
        family="masactrl",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_editor(build_masactrl_editor, "masactrl_tiny")
