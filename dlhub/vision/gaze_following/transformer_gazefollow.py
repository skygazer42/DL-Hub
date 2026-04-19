from __future__ import annotations

from torch import nn

from ._common import build_toy_gaze_follower, smoke_test_gaze_follower


_VARIANTS: dict[str, dict[str, int]] = {"transformer_gazefollow_tiny": {"width": 24, "depth": 1}, "transformer_gazefollow_small": {"width": 36, "depth": 2}, "transformer_gazefollow_base": {"width": 48, "depth": 3}}


def build_transformer_gazefollow_gaze_follower(*, in_channels: int, variant: str = "transformer_gazefollow_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_gaze_follower(family="transformer_gazefollow", mode="transformer", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_gaze_follower(build_transformer_gazefollow_gaze_follower, "transformer_gazefollow_tiny")
