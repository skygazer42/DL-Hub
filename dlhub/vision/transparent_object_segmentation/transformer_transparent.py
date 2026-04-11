from __future__ import annotations

from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "transformer_transparent_tiny": {"width": 24, "depth": 1},
    "transformer_transparent_small": {"width": 32, "depth": 2},
    "transformer_transparent_base": {"width": 48, "depth": 3},
}


def build_transformer_transparent_transparent_segmenter(
    *,
    in_channels: int,
    variant: str = "transformer_transparent_small",
    width_mult: float = 1.0,
    **kwargs,
):
    return build_toy_model(
        family="transformer_transparent",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(
        build_transformer_transparent_transparent_segmenter,
        "transformer_transparent_tiny",
    )
