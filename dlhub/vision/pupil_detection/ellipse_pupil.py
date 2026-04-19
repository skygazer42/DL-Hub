from __future__ import annotations

from torch import nn

from ._common import build_toy_pupil_detector, smoke_test_pupil_detector


_VARIANTS: dict[str, dict[str, int]] = {"ellipse_pupil_tiny": {"width": 24, "depth": 1}, "ellipse_pupil_small": {"width": 36, "depth": 2}, "ellipse_pupil_base": {"width": 48, "depth": 3}}


def build_ellipse_pupil_pupil_detector(*, in_channels: int, variant: str = "ellipse_pupil_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_pupil_detector(family="ellipse_pupil", mode="ellipse", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_pupil_detector(build_ellipse_pupil_pupil_detector, "ellipse_pupil_tiny")

