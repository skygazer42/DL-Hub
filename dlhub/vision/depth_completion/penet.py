from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "penet_tiny": {"width": 24, "depth": 1},
    "penet_small": {"width": 32, "depth": 2},
    "penet_base": {"width": 48, "depth": 3},
}


def build_penet_depth_completer(
    *, in_channels: int, variant: str = "penet_small", width_mult: float = 1.0
):
    return build_toy_model(
        family="penet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_penet_depth_completer, "penet_tiny")
