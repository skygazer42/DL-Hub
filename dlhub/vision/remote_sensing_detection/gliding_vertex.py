from __future__ import annotations
from torch import nn
from ._common import build_toy_rs_detector, smoke_test_rs

_VARIANTS = {
    "gliding_vertex_tiny": {"width": 24, "depth": 1},
    "gliding_vertex_small": {"width": 32, "depth": 2},
    "gliding_vertex_base": {"width": 48, "depth": 3},
}


def build_gliding_vertex_rs_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "gliding_vertex_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_rs_detector(
        family="gliding_vertex",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        oriented=True,
    )


if __name__ == "__main__":
    smoke_test_rs(build_gliding_vertex_rs_detector, "gliding_vertex_tiny")
