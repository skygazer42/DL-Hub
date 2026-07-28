from __future__ import annotations

from ._common import build_baseline_audio_visual_model, smoke_test_audio_visual_model

_VARIANTS = {
    "av_separation_tiny": {"width": 24, "depth": 1},
    "av_separation_small": {"width": 32, "depth": 2},
    "av_separation_base": {"width": 48, "depth": 3},
}


def build_av_separation_audio_visual_model(
    *,
    in_channels: int,
    variant: str = "av_separation_small",
    width_mult: float = 1.0,
    audio_bins: int = 32,
):
    return build_baseline_audio_visual_model(
        family="av_separation",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        audio_bins=int(audio_bins),
    )


if __name__ == "__main__":
    smoke_test_audio_visual_model(build_av_separation_audio_visual_model, "av_separation_tiny")
