from __future__ import annotations

from torch import nn

from ._common import build_baseline_vtr, smoke_test_vtr

_VARIANTS: dict[str, dict[str, int]] = {
    "dual_encoder_vtr_tiny": {"width": 24, "depth": 1},
    "dual_encoder_vtr_small": {"width": 32, "depth": 2},
    "dual_encoder_vtr_base": {"width": 48, "depth": 3},
}


def build_dual_encoder_vtr_retriever(
    *, in_channels: int, variant: str = "dual_encoder_vtr_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_vtr(
        family="dual_encoder_vtr",
        mode="dual",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vtr(build_dual_encoder_vtr_retriever, "dual_encoder_vtr_tiny")
