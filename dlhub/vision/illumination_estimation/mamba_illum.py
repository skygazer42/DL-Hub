from __future__ import annotations

from torch import nn

from ._common import build_baseline_illumination_estimator, smoke_test_illumination_estimator


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_illum_tiny": {"width": 24, "depth": 1},
    "mamba_illum_small": {"width": 36, "depth": 2},
    "mamba_illum_base": {"width": 48, "depth": 3},
}


def build_mamba_illum_illumination_estimator(
    *, in_channels: int, variant: str = "mamba_illum_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_illumination_estimator(
        family="mamba_illum",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_illumination_estimator(build_mamba_illum_illumination_estimator, "mamba_illum_tiny")
