from __future__ import annotations
from torch import nn
from ._common import build_baseline_open_vocabulary_3d_model, smoke_test_open_vocabulary_3d_model

_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_openvocab3d_tiny": {"width": 24, "depth": 1},
    "mamba_openvocab3d_small": {"width": 32, "depth": 2},
    "mamba_openvocab3d_base": {"width": 48, "depth": 3},
}


def build_mamba_openvocab3d_open_vocabulary_3d_model(
    *, in_channels: int, variant: str = "mamba_openvocab3d_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_open_vocabulary_3d_model(
        family="mamba_openvocab3d",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_open_vocabulary_3d_model(
        build_mamba_openvocab3d_open_vocabulary_3d_model, "mamba_openvocab3d_tiny"
    )
