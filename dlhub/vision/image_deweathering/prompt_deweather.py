from __future__ import annotations

from torch import nn

from ._common import build_toy_deweatherer, smoke_test_deweatherer


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_deweather_tiny": {"width": 24, "depth": 1, "passes": 1},
    "prompt_deweather_small": {"width": 32, "depth": 2, "passes": 2},
    "prompt_deweather_base": {"width": 48, "depth": 3, "passes": 2},
}


def build_prompt_deweather_deweatherer(
    *,
    in_channels: int,
    variant: str = "prompt_deweather_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_deweatherer(
        family="prompt_deweather",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_deweatherer(build_prompt_deweather_deweatherer, "prompt_deweather_tiny")
