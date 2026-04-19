from __future__ import annotations

from torch import nn

from ._common import build_toy_irstd_detector, smoke_test_irstd_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "tophat_irstd_tiny": {"width": 24, "depth": 1, "queries": 8},
    "tophat_irstd_small": {"width": 36, "depth": 2, "queries": 12},
    "tophat_irstd_base": {"width": 48, "depth": 3, "queries": 16},
}


def build_tophat_irstd_irstd_detector(
    *,
    in_channels: int,
    variant: str = "tophat_irstd_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_irstd_detector(
        family="tophat_irstd",
        mode="tophat",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_irstd_detector(build_tophat_irstd_irstd_detector, "tophat_irstd_tiny")

