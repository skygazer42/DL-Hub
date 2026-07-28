from __future__ import annotations

from torch import nn

from ._common import build_baseline_atu, smoke_test_atu

_VARIANTS: dict[str, dict[str, int]] = {
    "event_audio_text_tiny": {"width": 24, "depth": 1},
    "event_audio_text_small": {"width": 32, "depth": 2},
    "event_audio_text_base": {"width": 48, "depth": 3},
}


def build_event_audio_text_audio_text_model(
    *, in_channels: int = 1, variant: str = "event_audio_text_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_atu(
        family="event_audio_text",
        mode="event",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_atu(build_event_audio_text_audio_text_model, "event_audio_text_tiny")
