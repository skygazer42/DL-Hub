from __future__ import annotations

from torch import nn

from ._common import build_toy_stylizer, smoke_test_stylizer


_VARIANTS: dict[str, dict[str, int]] = {
    "ref_style_tiny": {"width": 24, "depth": 1},
    "ref_style_small": {"width": 36, "depth": 2},
    "ref_style_base": {"width": 48, "depth": 3},
}


def build_ref_style_stylizer(
    *,
    in_channels: int,
    variant: str = "ref_style_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_stylizer(
        family="ref_style",
        mode="dual",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_stylizer(build_ref_style_stylizer, "ref_style_tiny")
