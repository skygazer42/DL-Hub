from __future__ import annotations
from ._common import build_baseline_vos, smoke_test_vos

_VARIANTS = {
    "deaot_tiny": {"width": 24, "depth": 1},
    "deaot_small": {"width": 32, "depth": 2},
    "deaot_base": {"width": 48, "depth": 3},
}


def build_deaot_vos_model(
    *, in_channels: int, variant: str = "deaot_small", width_mult: float = 1.0, num_masks: int = 2
):
    return build_baseline_vos(
        family="deaot",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_masks=int(num_masks),
    )


if __name__ == "__main__":
    smoke_test_vos(build_deaot_vos_model, "deaot_tiny")
