from __future__ import annotations

from torch import nn

from ._common import build_baseline_ov_detector, smoke_test_ov_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_ovdet_tiny": {"width": 24, "depth": 1, "queries": 8, "vocab_size": 32},
    "mamba_ovdet_small": {"width": 36, "depth": 2, "queries": 12, "vocab_size": 48},
    "mamba_ovdet_base": {"width": 48, "depth": 3, "queries": 16, "vocab_size": 64},
}


def build_mamba_ovdet_detector(
    *,
    in_channels: int,
    variant: str = "mamba_ovdet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_ov_detector(
        family="mamba_ovdet",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ov_detector(build_mamba_ovdet_detector, "mamba_ovdet_tiny")
