from __future__ import annotations
from ._common import build_baseline_attr, smoke_test_attr

_VARIANTS = {
    "hydraplus_ped_tiny": {"width": 24, "depth": 1},
    "hydraplus_ped_small": {"width": 32, "depth": 2},
    "hydraplus_ped_base": {"width": 48, "depth": 3},
}


def build_hydraplus_ped_(
    *,
    in_channels: int,
    num_attributes: int,
    variant: str = "hydraplus_ped_small",
    width_mult: float = 1.0,
):
    return build_baseline_attr(
        family="hydraplus_ped",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_attributes=int(num_attributes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_attr(build_hydraplus_ped_, "hydraplus_ped_tiny")
