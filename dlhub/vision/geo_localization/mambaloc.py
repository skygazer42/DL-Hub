from __future__ import annotations
from ._common import build_toy_geo, smoke_test_geo

_VARIANTS = {
    "mambaloc_tiny": {"width": 24, "depth": 1, "embed": 128},
    "mambaloc_small": {"width": 32, "depth": 2, "embed": 160},
    "mambaloc_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_mambaloc_geo_localizer(
    *, in_channels: int, variant: str = "mambaloc_small", width_mult: float = 1.0
):
    return build_toy_geo(
        family="mambaloc",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_geo(build_mambaloc_geo_localizer, "mambaloc_tiny")
