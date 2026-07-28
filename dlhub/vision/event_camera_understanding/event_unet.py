from __future__ import annotations

from ._common import build_baseline_event_model, smoke_test_event_model

_VARIANTS = {
    "event_unet_tiny": {"width": 24, "depth": 1, "steps": 3},
    "event_unet_small": {"width": 32, "depth": 2, "steps": 4},
    "event_unet_base": {"width": 48, "depth": 3, "steps": 5},
}


def build_event_unet_event_model(
    *, in_channels: int, variant: str = "event_unet_small", width_mult: float = 1.0
):
    return build_baseline_event_model(
        family="event_unet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_event_model(build_event_unet_event_model, "event_unet_tiny")
