from __future__ import annotations
from ._common import build_baseline_augmenter, smoke_test_augmenter

_VARIANTS = {
    "mixup_aug_tiny": {"width": 24, "depth": 1},
    "mixup_aug_small": {"width": 32, "depth": 2},
    "mixup_aug_base": {"width": 48, "depth": 3},
}


def build_mixup_aug_augmenter(
    *, in_channels: int, variant: str = "mixup_aug_small", width_mult: float = 1.0
):
    return build_baseline_augmenter(
        family="mixup_aug",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_augmenter(build_mixup_aug_augmenter, "mixup_aug_tiny")
