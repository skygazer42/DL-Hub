from __future__ import annotations

from torch import nn

from ._common import build_toy_object_discoverer, smoke_test_object_discoverer


_VARIANTS: dict[str, dict[str, int]] = {
    "slot_objdisc_tiny": {"width": 24, "depth": 1, "slots": 6},
    "slot_objdisc_small": {"width": 36, "depth": 2, "slots": 8},
    "slot_objdisc_base": {"width": 48, "depth": 3, "slots": 10},
}


def build_slot_objdisc_object_discoverer(
    *, in_channels: int, variant: str = "slot_objdisc_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_object_discoverer(
        family="slot_objdisc",
        mode="slot",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_object_discoverer(build_slot_objdisc_object_discoverer, "slot_objdisc_tiny")
