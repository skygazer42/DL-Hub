from __future__ import annotations

from torch import nn

from ._common import build_toy_outpainter, smoke_test_outpainter


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_outpaint_tiny": {"width": 24, "depth": 1},
    "prompt_outpaint_small": {"width": 36, "depth": 2},
    "prompt_outpaint_base": {"width": 48, "depth": 3},
}


def build_prompt_outpaint_outpainter(
    *, in_channels: int, variant: str = "prompt_outpaint_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_outpainter(
        family="prompt_outpaint",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_outpainter(build_prompt_outpaint_outpainter, "prompt_outpaint_tiny")
