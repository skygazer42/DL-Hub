from __future__ import annotations

from ._common import build_baseline_event_model, smoke_test_event_model

_VARIANTS = {
    "ev_cnn_tiny": {"width": 24, "depth": 1, "steps": 2},
    "ev_cnn_small": {"width": 32, "depth": 2, "steps": 3},
    "ev_cnn_base": {"width": 48, "depth": 3, "steps": 4},
}


def build_ev_cnn_event_model(
    *, in_channels: int, variant: str = "ev_cnn_small", width_mult: float = 1.0
):
    return build_baseline_event_model(
        family="ev_cnn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_event_model(build_ev_cnn_event_model, "ev_cnn_tiny")
