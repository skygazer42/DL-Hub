from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "meshed_memory_tiny": {"width": 24, "depth": 1},
    "meshed_memory_small": {"width": 32, "depth": 2},
    "meshed_memory_base": {"width": 48, "depth": 3},
}


def build_meshed_memory_image_captioner(
    *, in_channels: int, variant: str = "meshed_memory_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="meshed_memory",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_meshed_memory_image_captioner, "meshed_memory_tiny")
