from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "chartformer_tiny": {"width": 24, "depth": 1},
    "chartformer_small": {"width": 32, "depth": 2},
    "chartformer_base": {"width": 48, "depth": 3},
}


def build_chartformer_chart_understander(
    *, in_channels: int, variant: str = "chartformer_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="chartformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_chartformer_chart_understander, "chartformer_tiny")
