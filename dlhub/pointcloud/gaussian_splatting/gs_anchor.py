from __future__ import annotations

from ._common import build_toy_splatter, smoke_test_splatter

_VARIANTS = {
    "gs_anchor_tiny": {"width": 24, "depth": 1},
    "gs_anchor_small": {"width": 32, "depth": 2},
    "gs_anchor_base": {"width": 48, "depth": 3},
}


def build_gs_anchor_splatter(
    *, in_channels: int, variant: str = "gs_anchor_small", width_mult: float = 1.0
):
    return build_toy_splatter(
        family="gs_anchor",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_splatter(build_gs_anchor_splatter, "gs_anchor_tiny")
