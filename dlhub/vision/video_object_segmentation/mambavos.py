from __future__ import annotations
from ._common import build_baseline_vos, smoke_test_vos

_VARIANTS = {
    "mambavos_tiny": {"width": 24, "depth": 1},
    "mambavos_small": {"width": 32, "depth": 2},
    "mambavos_base": {"width": 48, "depth": 3},
}


def build_mambavos_vos_model(
    *,
    in_channels: int,
    variant: str = "mambavos_small",
    width_mult: float = 1.0,
    num_masks: int = 2,
):
    return build_baseline_vos(
        family="mambavos",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_masks=int(num_masks),
    )


if __name__ == "__main__":
    smoke_test_vos(build_mambavos_vos_model, "mambavos_tiny")
