from __future__ import annotations
from ._common import build_baseline_attr, smoke_test_attr

_VARIANTS = {
    "attrcnn_tiny": {"width": 24, "depth": 1},
    "attrcnn_small": {"width": 32, "depth": 2},
    "attrcnn_base": {"width": 48, "depth": 3},
}


def build_attrcnn_attribute_recognizer(
    *,
    in_channels: int,
    num_attributes: int,
    variant: str = "attrcnn_small",
    width_mult: float = 1.0,
):
    return build_baseline_attr(
        family="attrcnn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_attributes=int(num_attributes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_attr(build_attrcnn_attribute_recognizer, "attrcnn_tiny")
