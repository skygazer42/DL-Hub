from __future__ import annotations

from torch import nn

from ._common import build_baseline_vtr, smoke_test_vtr

_VARIANTS: dict[str, dict[str, int]] = {
    "temporal_align_vtr_tiny": {"width": 24, "depth": 1},
    "temporal_align_vtr_small": {"width": 32, "depth": 2},
    "temporal_align_vtr_base": {"width": 48, "depth": 3},
}


def build_temporal_align_vtr_retriever(
    *, in_channels: int, variant: str = "temporal_align_vtr_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_vtr(
        family="temporal_align_vtr",
        mode="temporal",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vtr(build_temporal_align_vtr_retriever, "temporal_align_vtr_tiny")
