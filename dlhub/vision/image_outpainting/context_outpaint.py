from __future__ import annotations

from torch import nn

from ._common import build_baseline_outpainter, smoke_test_outpainter


_VARIANTS: dict[str, dict[str, int]] = {
    "context_outpaint_tiny": {"width": 24, "depth": 1},
    "context_outpaint_small": {"width": 36, "depth": 2},
    "context_outpaint_base": {"width": 48, "depth": 3},
}


def build_context_outpaint_outpainter(
    *, in_channels: int, variant: str = "context_outpaint_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_outpainter(
        family="context_outpaint",
        mode="context",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_outpainter(build_context_outpaint_outpainter, "context_outpaint_tiny")
