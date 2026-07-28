from __future__ import annotations

from ._common import build_baseline_event_model, smoke_test_event_model

_VARIANTS = {
    "mamba_event_tiny": {"width": 24, "depth": 1, "steps": 2},
    "mamba_event_small": {"width": 32, "depth": 2, "steps": 3},
    "mamba_event_base": {"width": 48, "depth": 3, "steps": 4},
}


def build_mamba_event_event_model(
    *, in_channels: int, variant: str = "mamba_event_small", width_mult: float = 1.0
):
    return build_baseline_event_model(
        family="mamba_event",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_event_model(build_mamba_event_event_model, "mamba_event_tiny")
