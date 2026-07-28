from __future__ import annotations

from torch import nn

from ._common import build_baseline_ov_detector, smoke_test_ov_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "detic_ovdet_tiny": {"width": 24, "depth": 1, "queries": 8, "vocab_size": 32},
    "detic_ovdet_small": {"width": 36, "depth": 2, "queries": 12, "vocab_size": 48},
    "detic_ovdet_base": {"width": 48, "depth": 3, "queries": 16, "vocab_size": 64},
}


def build_detic_ovdet_detector(
    *,
    in_channels: int,
    variant: str = "detic_ovdet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_ov_detector(
        family="detic_ovdet",
        mode="detic",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ov_detector(build_detic_ovdet_detector, "detic_ovdet_tiny")
