from __future__ import annotations

from torch import nn

from ._common import build_baseline_irstd_detector, smoke_test_irstd_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_irstd_tiny": {"width": 24, "depth": 1, "queries": 8},
    "transformer_irstd_small": {"width": 36, "depth": 2, "queries": 12},
    "transformer_irstd_base": {"width": 48, "depth": 3, "queries": 16},
}


def build_transformer_irstd_irstd_detector(
    *,
    in_channels: int,
    variant: str = "transformer_irstd_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_irstd_detector(
        family="transformer_irstd",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_irstd_detector(build_transformer_irstd_irstd_detector, "transformer_irstd_tiny")
