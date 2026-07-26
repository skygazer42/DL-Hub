from __future__ import annotations
from ._common import build_toy_expr, smoke_test_expr

_VARIANTS = {
    "cmcnn_expr_tiny": {"width": 24, "depth": 1},
    "cmcnn_expr_small": {"width": 32, "depth": 2},
    "cmcnn_expr_base": {"width": 48, "depth": 3},
}


def build_cmcnn_expr_expression_recognizer(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "cmcnn_expr_small",
    width_mult: float = 1.0,
):
    return build_toy_expr(
        family="cmcnn_expr",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_expr(build_cmcnn_expr_expression_recognizer, "cmcnn_expr_tiny")
