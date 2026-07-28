from __future__ import annotations
from ._common import build_baseline_geo, smoke_test_geo

_VARIANTS = {
    "cvm_net_tiny": {"width": 24, "depth": 1, "embed": 128},
    "cvm_net_small": {"width": 32, "depth": 2, "embed": 160},
    "cvm_net_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_cvm_net_geo_localizer(
    *, in_channels: int, variant: str = "cvm_net_small", width_mult: float = 1.0
):
    return build_baseline_geo(
        family="cvm_net",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_geo(build_cvm_net_geo_localizer, "cvm_net_tiny")
