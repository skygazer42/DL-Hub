from __future__ import annotations

from torch import nn

from ._common import build_baseline_gaze_follower, smoke_test_gaze_follower


_VARIANTS: dict[str, dict[str, int]] = {
    "coarse_gazefollow_tiny": {"width": 24, "depth": 1},
    "coarse_gazefollow_small": {"width": 36, "depth": 2},
    "coarse_gazefollow_base": {"width": 48, "depth": 3},
}


def build_coarse_gazefollow_gaze_follower(
    *, in_channels: int, variant: str = "coarse_gazefollow_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_gaze_follower(
        family="coarse_gazefollow",
        mode="coarse",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_gaze_follower(build_coarse_gazefollow_gaze_follower, "coarse_gazefollow_tiny")
