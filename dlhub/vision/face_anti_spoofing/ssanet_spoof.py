from __future__ import annotations
from ._common import build_baseline_spoofer, smoke_test_spoof

_VARIANTS = {
    "ssanet_spoof_tiny": {"width": 24, "depth": 1},
    "ssanet_spoof_small": {"width": 32, "depth": 2},
    "ssanet_spoof_base": {"width": 48, "depth": 3},
}


def build_ssanet_spoof_anti_spoofer(
    *, in_channels: int, variant: str = "ssanet_spoof_small", width_mult: float = 1.0
):
    return build_baseline_spoofer(
        family="ssanet_spoof",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_spoof(build_ssanet_spoof_anti_spoofer, "ssanet_spoof_tiny")
