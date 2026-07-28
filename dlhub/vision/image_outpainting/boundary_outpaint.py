from __future__ import annotations

from torch import nn

from ._common import build_baseline_outpainter, smoke_test_outpainter


_VARIANTS: dict[str, dict[str, int]] = {
    "boundary_outpaint_tiny": {"width": 24, "depth": 1},
    "boundary_outpaint_small": {"width": 36, "depth": 2},
    "boundary_outpaint_base": {"width": 48, "depth": 3},
}


def build_boundary_outpaint_outpainter(
    *, in_channels: int, variant: str = "boundary_outpaint_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_outpainter(
        family="boundary_outpaint",
        mode="boundary",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_outpainter(build_boundary_outpaint_outpainter, "boundary_outpaint_tiny")
