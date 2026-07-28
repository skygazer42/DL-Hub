from __future__ import annotations
from torch import nn
from ._common import build_baseline_shape_correspondence_model, smoke_test_shape_correspondence_model

_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_corr3d_tiny": {"width": 24, "depth": 1},
    "transformer_corr3d_small": {"width": 32, "depth": 2},
    "transformer_corr3d_base": {"width": 48, "depth": 3},
}


def build_transformer_corr3d_shape_correspondence_model(
    *, in_channels: int, variant: str = "transformer_corr3d_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_shape_correspondence_model(
        family="transformer_corr3d",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_shape_correspondence_model(
        build_transformer_corr3d_shape_correspondence_model, "transformer_corr3d_tiny"
    )
