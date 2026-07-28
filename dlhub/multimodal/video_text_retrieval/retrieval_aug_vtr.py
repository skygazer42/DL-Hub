from __future__ import annotations

from torch import nn

from ._common import build_baseline_vtr, smoke_test_vtr

_VARIANTS: dict[str, dict[str, int]] = {
    "retrieval_aug_vtr_tiny": {"width": 24, "depth": 1},
    "retrieval_aug_vtr_small": {"width": 32, "depth": 2},
    "retrieval_aug_vtr_base": {"width": 48, "depth": 3},
}


def build_retrieval_aug_vtr_retriever(
    *, in_channels: int, variant: str = "retrieval_aug_vtr_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_vtr(
        family="retrieval_aug_vtr",
        mode="retrieval_aug",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vtr(build_retrieval_aug_vtr_retriever, "retrieval_aug_vtr_tiny")
