from __future__ import annotations
from ._common import build_baseline_augmenter, smoke_test_augmenter

_VARIANTS = {
    "autoaugment_aug_tiny": {"width": 24, "depth": 1},
    "autoaugment_aug_small": {"width": 32, "depth": 2},
    "autoaugment_aug_base": {"width": 48, "depth": 3},
}


def build_autoaugment_aug_augmenter(
    *, in_channels: int, variant: str = "autoaugment_aug_small", width_mult: float = 1.0
):
    return build_baseline_augmenter(
        family="autoaugment_aug",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_augmenter(build_autoaugment_aug_augmenter, "autoaugment_aug_tiny")
