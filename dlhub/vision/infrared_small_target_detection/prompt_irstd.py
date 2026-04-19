from __future__ import annotations

from torch import nn

from ._common import build_toy_irstd_detector, smoke_test_irstd_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_irstd_tiny": {"width": 24, "depth": 1, "queries": 8},
    "prompt_irstd_small": {"width": 36, "depth": 2, "queries": 12},
    "prompt_irstd_base": {"width": 48, "depth": 3, "queries": 16},
}


def build_prompt_irstd_irstd_detector(
    *,
    in_channels: int,
    variant: str = "prompt_irstd_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_irstd_detector(
        family="prompt_irstd",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_irstd_detector(build_prompt_irstd_irstd_detector, "prompt_irstd_tiny")

