from __future__ import annotations

from ._common import build_toy_event_model, smoke_test_event_model

_VARIANTS = {
    "transformer_event_tiny": {"width": 24, "depth": 1, "steps": 2},
    "transformer_event_small": {"width": 32, "depth": 2, "steps": 3},
    "transformer_event_base": {"width": 48, "depth": 3, "steps": 4},
}


def build_transformer_event_event_model(
    *, in_channels: int, variant: str = "transformer_event_small", width_mult: float = 1.0
):
    return build_toy_event_model(
        family="transformer_event",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_event_model(build_transformer_event_event_model, "transformer_event_tiny")
