from __future__ import annotations

from torch import nn

from ._common import build_toy_translator, smoke_test_translator


_VARIANTS: dict[str, dict[str, int]] = {
    "pix2pix_translation_tiny": {"width": 24, "depth": 1},
    "pix2pix_translation_small": {"width": 36, "depth": 2},
    "pix2pix_translation_base": {"width": 48, "depth": 3},
}


def build_pix2pix_translation_translator(
    *,
    in_channels: int,
    variant: str = "pix2pix_translation_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_translator(
        family="pix2pix_translation",
        mode="pix2pix",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_translator(build_pix2pix_translation_translator, "pix2pix_translation_tiny")
