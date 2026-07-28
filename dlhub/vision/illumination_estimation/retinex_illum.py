from __future__ import annotations

from torch import nn

from ._common import build_baseline_illumination_estimator, smoke_test_illumination_estimator


_VARIANTS: dict[str, dict[str, int]] = {
    "retinex_illum_tiny": {"width": 24, "depth": 1},
    "retinex_illum_small": {"width": 36, "depth": 2},
    "retinex_illum_base": {"width": 48, "depth": 3},
}


def build_retinex_illum_illumination_estimator(
    *, in_channels: int, variant: str = "retinex_illum_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_illumination_estimator(
        family="retinex_illum",
        mode="retinex",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_illumination_estimator(
        build_retinex_illum_illumination_estimator, "retinex_illum_tiny"
    )
