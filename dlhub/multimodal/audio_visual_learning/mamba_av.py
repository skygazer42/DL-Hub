from __future__ import annotations

from ._common import build_toy_audio_visual_model, smoke_test_audio_visual_model

_VARIANTS = {
    "mamba_av_tiny": {"width": 24, "depth": 1},
    "mamba_av_small": {"width": 32, "depth": 2},
    "mamba_av_base": {"width": 48, "depth": 3},
}


def build_mamba_av_audio_visual_model(
    *, in_channels: int, variant: str = "mamba_av_small", width_mult: float = 1.0, audio_bins: int = 32
):
    return build_toy_audio_visual_model(
        family="mamba_av",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        audio_bins=int(audio_bins),
    )


if __name__ == "__main__":
    smoke_test_audio_visual_model(build_mamba_av_audio_visual_model, "mamba_av_tiny")
