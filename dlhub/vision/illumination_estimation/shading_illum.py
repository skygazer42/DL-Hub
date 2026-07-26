from __future__ import annotations

from torch import nn

from ._common import build_toy_illumination_estimator, smoke_test_illumination_estimator


_VARIANTS: dict[str, dict[str, int]] = {
    "shading_illum_tiny": {"width": 24, "depth": 1},
    "shading_illum_small": {"width": 36, "depth": 2},
    "shading_illum_base": {"width": 48, "depth": 3},
}


def build_shading_illum_illumination_estimator(
    *, in_channels: int, variant: str = "shading_illum_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_illumination_estimator(
        family="shading_illum",
        mode="shading",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_illumination_estimator(
        build_shading_illum_illumination_estimator, "shading_illum_tiny"
    )
