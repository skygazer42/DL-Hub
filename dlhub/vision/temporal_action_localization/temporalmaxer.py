from __future__ import annotations
from ._common import build_temporal_action_localization_baseline, smoke_test_model

_VARIANTS = {
    "temporalmaxer_tiny": {"width": 24, "depth": 1},
    "temporalmaxer_small": {"width": 32, "depth": 2},
    "temporalmaxer_base": {"width": 48, "depth": 3},
}


def build_temporalmaxer_tal_model(
    *, in_channels: int, variant: str = "temporalmaxer_small", width_mult: float = 1.0, **kwargs
):
    return build_temporal_action_localization_baseline(
        registered_alias="temporalmaxer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_temporalmaxer_tal_model, "temporalmaxer_tiny")
