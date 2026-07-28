from __future__ import annotations

from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "prompt_transparent_tiny": {"width": 24, "depth": 1},
    "prompt_transparent_small": {"width": 32, "depth": 2},
    "prompt_transparent_base": {"width": 48, "depth": 3},
}


def build_prompt_transparent_transparent_segmenter(
    *,
    in_channels: int,
    variant: str = "prompt_transparent_small",
    width_mult: float = 1.0,
    **kwargs,
):
    return build_baseline_model(
        family="prompt_transparent",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_prompt_transparent_transparent_segmenter, "prompt_transparent_tiny")
