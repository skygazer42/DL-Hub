from __future__ import annotations
from ._common import build_toy_change, smoke_test_change

_VARIANTS = {
    "ifn_cd_tiny": {"width": 24, "depth": 1},
    "ifn_cd_small": {"width": 32, "depth": 2},
    "ifn_cd_base": {"width": 48, "depth": 3},
}


def build_ifn_cd_change_detector(
    *, in_channels: int, variant: str = "ifn_cd_small", width_mult: float = 1.0
):
    return build_toy_change(
        family="ifn_cd",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_change(build_ifn_cd_change_detector, "ifn_cd_tiny")
