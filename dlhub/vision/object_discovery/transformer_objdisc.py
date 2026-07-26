from __future__ import annotations

from torch import nn

from ._common import build_toy_object_discoverer, smoke_test_object_discoverer


_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_objdisc_tiny": {"width": 24, "depth": 1, "slots": 6},
    "transformer_objdisc_small": {"width": 36, "depth": 2, "slots": 8},
    "transformer_objdisc_base": {"width": 48, "depth": 3, "slots": 10},
}


def build_transformer_objdisc_object_discoverer(
    *, in_channels: int, variant: str = "transformer_objdisc_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_object_discoverer(
        family="transformer_objdisc",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_object_discoverer(
        build_transformer_objdisc_object_discoverer, "transformer_objdisc_tiny"
    )
