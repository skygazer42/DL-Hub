from __future__ import annotations

from torch import nn

from ._common import build_toy_drop_remover, smoke_test_drop_remover


_VARIANTS: dict[str, dict[str, int]] = {
    "texture_refine_drop_tiny": {"width": 24, "depth": 1, "steps": 1},
    "texture_refine_drop_small": {"width": 32, "depth": 2, "steps": 2},
    "texture_refine_drop_base": {"width": 48, "depth": 3, "steps": 2},
}


def build_texture_refine_drop_drop_remover(
    *,
    in_channels: int,
    variant: str = "texture_refine_drop_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_drop_remover(
        family="texture_refine_drop",
        mode="texture_refine",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_drop_remover(build_texture_refine_drop_drop_remover, "texture_refine_drop_tiny")
