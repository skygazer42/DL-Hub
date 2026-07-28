from __future__ import annotations

from torch import nn

from ._common import build_baseline_outpainter, smoke_test_outpainter


_VARIANTS: dict[str, dict[str, int]] = {
    "cnn_outpaint_tiny": {"width": 24, "depth": 1},
    "cnn_outpaint_small": {"width": 36, "depth": 2},
    "cnn_outpaint_base": {"width": 48, "depth": 3},
}


def build_cnn_outpaint_outpainter(
    *, in_channels: int, variant: str = "cnn_outpaint_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_outpainter(
        family="cnn_outpaint",
        mode="cnn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_outpainter(build_cnn_outpaint_outpainter, "cnn_outpaint_tiny")
