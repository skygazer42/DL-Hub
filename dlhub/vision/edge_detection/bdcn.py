from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "bdcn_tiny": {"width": 24, "depth": 1},
    "bdcn_small": {"width": 32, "depth": 2},
    "bdcn_base": {"width": 48, "depth": 3},
}


def build_bdcn_edge_detector(
    *, in_channels: int, variant: str = "bdcn_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="bdcn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_bdcn_edge_detector, "bdcn_tiny")
