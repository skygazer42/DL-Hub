from __future__ import annotations
from ._common import build_toy_change, smoke_test_change

_VARIANTS = {
    "snunet_cd_tiny": {"width": 24, "depth": 1},
    "snunet_cd_small": {"width": 32, "depth": 2},
    "snunet_cd_base": {"width": 48, "depth": 3},
}


def build_snunet_cd_change_detector(
    *, in_channels: int, variant: str = "snunet_cd_small", width_mult: float = 1.0
):
    return build_toy_change(
        family="snunet_cd",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_change(build_snunet_cd_change_detector, "snunet_cd_tiny")
