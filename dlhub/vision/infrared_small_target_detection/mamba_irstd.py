from __future__ import annotations

from torch import nn

from ._common import build_baseline_irstd_detector, smoke_test_irstd_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_irstd_tiny": {"width": 24, "depth": 1, "queries": 8},
    "mamba_irstd_small": {"width": 36, "depth": 2, "queries": 12},
    "mamba_irstd_base": {"width": 48, "depth": 3, "queries": 16},
}


def build_mamba_irstd_irstd_detector(
    *,
    in_channels: int,
    variant: str = "mamba_irstd_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_irstd_detector(
        family="mamba_irstd",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_irstd_detector(build_mamba_irstd_irstd_detector, "mamba_irstd_tiny")
