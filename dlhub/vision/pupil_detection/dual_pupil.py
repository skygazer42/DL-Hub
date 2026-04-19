from __future__ import annotations

from torch import nn

from ._common import build_toy_pupil_detector, smoke_test_pupil_detector


_VARIANTS: dict[str, dict[str, int]] = {"dual_pupil_tiny": {"width": 24, "depth": 1}, "dual_pupil_small": {"width": 36, "depth": 2}, "dual_pupil_base": {"width": 48, "depth": 3}}


def build_dual_pupil_pupil_detector(*, in_channels: int, variant: str = "dual_pupil_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_pupil_detector(family="dual_pupil", mode="dual", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_pupil_detector(build_dual_pupil_pupil_detector, "dual_pupil_tiny")

