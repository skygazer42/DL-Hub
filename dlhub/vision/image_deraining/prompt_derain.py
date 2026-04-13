from __future__ import annotations

from torch import nn

from ._common import build_toy_derainer, smoke_test_derainer


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_derain_tiny": {"width": 24, "depth": 1, "steps": 1},
    "prompt_derain_small": {"width": 32, "depth": 2, "steps": 2},
    "prompt_derain_base": {"width": 48, "depth": 3, "steps": 2},
}


def build_prompt_derain_derainer(
    *,
    in_channels: int,
    variant: str = "prompt_derain_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_derainer(
        family="prompt_derain",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_derainer(build_prompt_derain_derainer, "prompt_derain_tiny")
