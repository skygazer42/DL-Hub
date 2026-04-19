from __future__ import annotations

from torch import nn

from ._common import build_toy_homography_estimator, smoke_test_homography_estimator


_VARIANTS: dict[str, dict[str, int]] = {"transformer_homography_tiny": {"width": 24, "depth": 1}, "transformer_homography_small": {"width": 36, "depth": 2}, "transformer_homography_base": {"width": 48, "depth": 3}}


def build_transformer_homography_homography_estimator(*, in_channels: int, variant: str = "transformer_homography_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_homography_estimator(family="transformer_homography", mode="transformer", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_homography_estimator(build_transformer_homography_homography_estimator, "transformer_homography_tiny")
