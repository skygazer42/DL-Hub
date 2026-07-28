from __future__ import annotations

from torch import nn

from ._common import build_baseline_deweatherer, smoke_test_deweatherer


_VARIANTS: dict[str, dict[str, int]] = {
    "deweather_cnn_tiny": {"width": 24, "depth": 1, "passes": 1},
    "deweather_cnn_small": {"width": 32, "depth": 2, "passes": 2},
    "deweather_cnn_base": {"width": 48, "depth": 3, "passes": 2},
}


def build_deweather_cnn_deweatherer(
    *,
    in_channels: int,
    variant: str = "deweather_cnn_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_deweatherer(
        family="deweather_cnn",
        mode="cnn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_deweatherer(build_deweather_cnn_deweatherer, "deweather_cnn_tiny")
