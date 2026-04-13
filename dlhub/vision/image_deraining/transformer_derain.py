from __future__ import annotations

from torch import nn

from ._common import build_toy_derainer, smoke_test_derainer


_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_derain_tiny": {"width": 24, "depth": 1, "steps": 1},
    "transformer_derain_small": {"width": 32, "depth": 2, "steps": 2},
    "transformer_derain_base": {"width": 48, "depth": 3, "steps": 2},
}


def build_transformer_derain_derainer(
    *,
    in_channels: int,
    variant: str = "transformer_derain_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_derainer(
        family="transformer_derain",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_derainer(build_transformer_derain_derainer, "transformer_derain_tiny")
