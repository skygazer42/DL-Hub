from __future__ import annotations

from torch import nn

from ._common import build_baseline_crowd_localizer, smoke_test_crowd_localizer


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_crowdloc_tiny": {"width": 24, "depth": 1, "num_points": 6},
    "mamba_crowdloc_small": {"width": 36, "depth": 2, "num_points": 8},
    "mamba_crowdloc_base": {"width": 48, "depth": 3, "num_points": 10},
}


def build_mamba_crowdloc_crowd_localizer(
    *, in_channels: int, variant: str = "mamba_crowdloc_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_crowd_localizer(
        family="mamba_crowdloc",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_crowd_localizer(build_mamba_crowdloc_crowd_localizer, "mamba_crowdloc_tiny")
