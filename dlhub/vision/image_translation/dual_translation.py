from __future__ import annotations

from torch import nn

from ._common import build_baseline_translator, smoke_test_translator


_VARIANTS: dict[str, dict[str, int]] = {
    "dual_translation_tiny": {"width": 24, "depth": 1},
    "dual_translation_small": {"width": 36, "depth": 2},
    "dual_translation_base": {"width": 48, "depth": 3},
}


def build_dual_translation_translator(
    *,
    in_channels: int,
    variant: str = "dual_translation_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_translator(
        family="dual_translation",
        mode="dual",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_translator(build_dual_translation_translator, "dual_translation_tiny")
