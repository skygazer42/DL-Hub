from __future__ import annotations

from torch import nn

from ._common import build_baseline_derainer, smoke_test_derainer


_VARIANTS: dict[str, dict[str, int]] = {
    "frequency_derain_tiny": {"width": 24, "depth": 1, "steps": 1},
    "frequency_derain_small": {"width": 32, "depth": 2, "steps": 1},
    "frequency_derain_base": {"width": 48, "depth": 3, "steps": 2},
}


def build_frequency_derain_derainer(
    *,
    in_channels: int,
    variant: str = "frequency_derain_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_derainer(
        family="frequency_derain",
        mode="frequency",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_derainer(build_frequency_derain_derainer, "frequency_derain_tiny")
