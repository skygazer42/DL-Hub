from __future__ import annotations

from torch import nn

from ._common import build_toy_atu, smoke_test_atu

_VARIANTS: dict[str, dict[str, int]] = {
    "contrastive_atu_tiny": {"width": 24, "depth": 1},
    "contrastive_atu_small": {"width": 32, "depth": 2},
    "contrastive_atu_base": {"width": 48, "depth": 3},
}


def build_contrastive_atu_audio_text_model(
    *, in_channels: int = 1, variant: str = "contrastive_atu_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_atu(
        family="contrastive_atu",
        mode="contrastive",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_atu(build_contrastive_atu_audio_text_model, "contrastive_atu_tiny")
