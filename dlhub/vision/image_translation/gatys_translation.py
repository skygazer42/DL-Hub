from __future__ import annotations

from torch import nn

from ._common import build_toy_translator, smoke_test_translator


_VARIANTS: dict[str, dict[str, int]] = {
    "gatys_translation_tiny": {"width": 24, "depth": 1},
    "gatys_translation_small": {"width": 36, "depth": 2},
    "gatys_translation_base": {"width": 48, "depth": 3},
}


def build_gatys_translation_translator(
    *,
    in_channels: int,
    variant: str = "gatys_translation_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_translator(
        family="gatys_translation",
        mode="gatys",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_translator(build_gatys_translation_translator, "gatys_translation_tiny")
