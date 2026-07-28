from __future__ import annotations
from ._common import build_baseline_geo, smoke_test_geo

_VARIANTS = {
    "streetclip_tiny": {"width": 24, "depth": 1, "embed": 128},
    "streetclip_small": {"width": 32, "depth": 2, "embed": 160},
    "streetclip_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_streetclip_geo_localizer(
    *, in_channels: int, variant: str = "streetclip_small", width_mult: float = 1.0
):
    return build_baseline_geo(
        family="streetclip",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_geo(build_streetclip_geo_localizer, "streetclip_tiny")
