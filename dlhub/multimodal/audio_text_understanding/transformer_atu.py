from __future__ import annotations

from torch import nn

from ._common import build_toy_atu, smoke_test_atu

_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_atu_tiny": {"width": 24, "depth": 1},
    "transformer_atu_small": {"width": 32, "depth": 2},
    "transformer_atu_base": {"width": 48, "depth": 3},
}


def build_transformer_atu_audio_text_model(
    *, in_channels: int = 1, variant: str = "transformer_atu_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_atu(
        family="transformer_atu",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_atu(build_transformer_atu_audio_text_model, "transformer_atu_tiny")
