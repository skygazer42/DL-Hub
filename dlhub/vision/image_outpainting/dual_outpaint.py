from __future__ import annotations

from torch import nn

from ._common import build_toy_outpainter, smoke_test_outpainter


_VARIANTS: dict[str, dict[str, int]] = {
    "dual_outpaint_tiny": {"width": 24, "depth": 1},
    "dual_outpaint_small": {"width": 36, "depth": 2},
    "dual_outpaint_base": {"width": 48, "depth": 3},
}


def build_dual_outpaint_outpainter(
    *, in_channels: int, variant: str = "dual_outpaint_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_outpainter(
        family="dual_outpaint",
        mode="dual",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_outpainter(build_dual_outpaint_outpainter, "dual_outpaint_tiny")
