from __future__ import annotations

from torch import nn

from ._common import build_toy_gaze_follower, smoke_test_gaze_follower


_VARIANTS: dict[str, dict[str, int]] = {
    "target_gazefollow_tiny": {"width": 24, "depth": 1},
    "target_gazefollow_small": {"width": 36, "depth": 2},
    "target_gazefollow_base": {"width": 48, "depth": 3},
}


def build_target_gazefollow_gaze_follower(
    *, in_channels: int, variant: str = "target_gazefollow_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_gaze_follower(
        family="target_gazefollow",
        mode="target",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_gaze_follower(build_target_gazefollow_gaze_follower, "target_gazefollow_tiny")
