from __future__ import annotations

from torch import nn

from ._common import build_toy_motion_segmentor, smoke_test_motion_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_motionseg_tiny": {"width": 24, "depth": 1},
    "prompt_motionseg_small": {"width": 36, "depth": 2},
    "prompt_motionseg_base": {"width": 48, "depth": 3},
}


def build_prompt_motionseg_motion_segmentor(
    *, in_channels: int, variant: str = "prompt_motionseg_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_motion_segmentor(
        family="prompt_motionseg",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_motion_segmentor(build_prompt_motionseg_motion_segmentor, "prompt_motionseg_tiny")
