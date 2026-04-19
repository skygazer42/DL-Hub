from __future__ import annotations

from torch import nn

from ._common import build_toy_outpainter, smoke_test_outpainter


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_outpaint_tiny": {"width": 24, "depth": 1},
    "mamba_outpaint_small": {"width": 36, "depth": 2},
    "mamba_outpaint_base": {"width": 48, "depth": 3},
}


def build_mamba_outpaint_outpainter(
    *, in_channels: int, variant: str = "mamba_outpaint_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_outpainter(
        family="mamba_outpaint",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_outpainter(build_mamba_outpaint_outpainter, "mamba_outpaint_tiny")
