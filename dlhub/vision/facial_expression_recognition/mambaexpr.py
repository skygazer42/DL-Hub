from __future__ import annotations
from ._common import build_baseline_expr, smoke_test_expr

_VARIANTS = {
    "mambaexpr_tiny": {"width": 24, "depth": 1},
    "mambaexpr_small": {"width": 32, "depth": 2},
    "mambaexpr_base": {"width": 48, "depth": 3},
}


def build_mambaexpr_expression_recognizer(
    *, in_channels: int, num_classes: int, variant: str = "mambaexpr_small", width_mult: float = 1.0
):
    return build_baseline_expr(
        family="mambaexpr",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_expr(build_mambaexpr_expression_recognizer, "mambaexpr_tiny")
