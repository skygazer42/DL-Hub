from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "tabchart_tiny": {"width": 24, "depth": 1},
    "tabchart_small": {"width": 32, "depth": 2},
    "tabchart_base": {"width": 48, "depth": 3},
}


def build_tabchart_chart_understander(
    *, in_channels: int, variant: str = "tabchart_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="tabchart",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_tabchart_chart_understander, "tabchart_tiny")
