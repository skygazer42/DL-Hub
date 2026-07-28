from __future__ import annotations
from ._common import build_temporal_action_localization_baseline, smoke_test_model

_VARIANTS = {
    "bsn2_tiny": {"width": 24, "depth": 1},
    "bsn2_small": {"width": 32, "depth": 2},
    "bsn2_base": {"width": 48, "depth": 3},
}


def build_bsn2_tal_model(
    *, in_channels: int, variant: str = "bsn2_small", width_mult: float = 1.0, **kwargs
):
    return build_temporal_action_localization_baseline(
        registered_alias="bsn2",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_bsn2_tal_model, "bsn2_tiny")
