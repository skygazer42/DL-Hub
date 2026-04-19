from __future__ import annotations

from torch import nn

from ._common import build_toy_glare_detector, smoke_test_glare_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_glare_tiny": {"width": 24, "depth": 1, "bins": 6},
    "prompt_glare_small": {"width": 36, "depth": 2, "bins": 8},
    "prompt_glare_base": {"width": 48, "depth": 3, "bins": 12},
}


def build_prompt_glare_glare_detector(
    *,
    in_channels: int,
    variant: str = "prompt_glare_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_glare_detector(
        family="prompt_glare",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_glare_detector(build_prompt_glare_glare_detector, "prompt_glare_tiny")
