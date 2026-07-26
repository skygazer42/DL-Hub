from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "prompt_aes_tiny": {"width": 24, "depth": 1},
    "prompt_aes_small": {"width": 32, "depth": 2},
    "prompt_aes_base": {"width": 48, "depth": 3},
}


def build_prompt_aes_aesthetic_model(
    *, in_channels: int, variant: str = "prompt_aes_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="prompt_aes",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_prompt_aes_aesthetic_model, "prompt_aes_tiny")
