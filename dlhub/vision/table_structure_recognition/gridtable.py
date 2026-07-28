from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "gridtable_tiny": {"width": 24, "depth": 1},
    "gridtable_small": {"width": 32, "depth": 2},
    "gridtable_base": {"width": 48, "depth": 3},
}


def build_gridtable_table_parser(
    *, in_channels: int, variant: str = "gridtable_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="gridtable",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_gridtable_table_parser, "gridtable_tiny")
