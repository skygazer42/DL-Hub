from __future__ import annotations
from ._common import build_referring_expression_baseline, smoke_test_model

_VARIANTS = {
    "transvg_tiny": {"width": 24, "depth": 1},
    "transvg_small": {"width": 32, "depth": 2},
    "transvg_base": {"width": 48, "depth": 3},
}


def build_transvg_refexp_grounder(
    *, in_channels: int, variant: str = "transvg_small", width_mult: float = 1.0, **kwargs
):
    return build_referring_expression_baseline(
        registered_alias="transvg",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_transvg_refexp_grounder, "transvg_tiny")
