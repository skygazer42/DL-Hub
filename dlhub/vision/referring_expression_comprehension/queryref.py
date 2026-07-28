from __future__ import annotations
from ._common import build_referring_expression_baseline, smoke_test_model

_VARIANTS = {
    "queryref_tiny": {"width": 24, "depth": 1},
    "queryref_small": {"width": 32, "depth": 2},
    "queryref_base": {"width": 48, "depth": 3},
}


def build_queryref_refexp_grounder(
    *, in_channels: int, variant: str = "queryref_small", width_mult: float = 1.0, **kwargs
):
    return build_referring_expression_baseline(
        registered_alias="queryref",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_queryref_refexp_grounder, "queryref_tiny")
