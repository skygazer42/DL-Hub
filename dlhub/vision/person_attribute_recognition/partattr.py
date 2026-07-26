from __future__ import annotations
from ._common import build_toy_attr, smoke_test_attr

_VARIANTS = {
    "partattr_tiny": {"width": 24, "depth": 1},
    "partattr_small": {"width": 32, "depth": 2},
    "partattr_base": {"width": 48, "depth": 3},
}


def build_partattr_attribute_recognizer(
    *,
    in_channels: int,
    num_attributes: int,
    variant: str = "partattr_small",
    width_mult: float = 1.0,
):
    return build_toy_attr(
        family="partattr",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_attributes=int(num_attributes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_attr(build_partattr_attribute_recognizer, "partattr_tiny")
