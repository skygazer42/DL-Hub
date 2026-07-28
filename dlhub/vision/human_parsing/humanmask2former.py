from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "humanmask2former_tiny": {"width": 24, "depth": 1},
    "humanmask2former_small": {"width": 32, "depth": 2},
    "humanmask2former_base": {"width": 48, "depth": 3},
}


def build_humanmask2former_human_parser(
    *, in_channels: int, variant: str = "humanmask2former_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="humanmask2former",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_humanmask2former_human_parser, "humanmask2former_tiny")
