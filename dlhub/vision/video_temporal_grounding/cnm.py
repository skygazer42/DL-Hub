from __future__ import annotations
from ._common import build_temporal_grounding_baseline, smoke_test_model

_VARIANTS = {
    "cnm_tiny": {"width": 24, "depth": 1},
    "cnm_small": {"width": 32, "depth": 2},
    "cnm_base": {"width": 48, "depth": 3},
}


def build_cnm_temporal_grounder(
    *, in_channels: int, variant: str = "cnm_small", width_mult: float = 1.0, **kwargs
):
    return build_temporal_grounding_baseline(
        registered_alias="cnm",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_cnm_temporal_grounder, "cnm_tiny")
