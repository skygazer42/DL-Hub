from __future__ import annotations

from torch import nn

from ._common import build_toy_flare_remover, smoke_test_flare_remover


_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_flare_tiny": {"width": 24, "depth": 1, "steps": 1},
    "transformer_flare_small": {"width": 36, "depth": 2, "steps": 1},
    "transformer_flare_base": {"width": 48, "depth": 3, "steps": 2},
}


def build_transformer_flare_flare_remover(
    *, in_channels: int, variant: str = "transformer_flare_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_flare_remover(
        family="transformer_flare",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_flare_remover(build_transformer_flare_flare_remover, "transformer_flare_tiny")
