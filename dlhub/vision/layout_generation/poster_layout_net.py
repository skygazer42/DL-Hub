from __future__ import annotations

from ._common import build_compact_layout_generator, validate_layout_generator

_VARIANTS = {
    "poster_layout_net_tiny": {"width": 24, "depth": 1},
    "poster_layout_net_small": {"width": 32, "depth": 2},
    "poster_layout_net_base": {"width": 48, "depth": 3},
}


def build_poster_layout_net_layout_generator(
    *, in_channels: int, variant: str = "poster_layout_net_small", width_mult: float = 1.0
):
    return build_compact_layout_generator(
        family="poster_layout_net",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    validate_layout_generator(
        build_poster_layout_net_layout_generator, "poster_layout_net_tiny"
    )
