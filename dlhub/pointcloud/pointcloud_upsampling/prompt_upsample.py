from __future__ import annotations
from torch import nn
from ._common import build_baseline_upsampler, smoke_test_upsampler

_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_upsample_tiny": {"width": 24, "depth": 1, "up_factor": 2},
    "prompt_upsample_small": {"width": 32, "depth": 2, "up_factor": 2},
    "prompt_upsample_base": {"width": 48, "depth": 3, "up_factor": 4},
}


def build_prompt_upsample_upsampler(
    *, in_channels: int, variant: str = "prompt_upsample_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_upsampler(
        family="prompt_upsample",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_upsampler(build_prompt_upsample_upsampler, "prompt_upsample_tiny")
