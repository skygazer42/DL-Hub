from __future__ import annotations
from torch import nn
from ._common import build_toy_open_vocabulary_3d_model, smoke_test_open_vocabulary_3d_model

_VARIANTS: dict[str, dict[str, int]] = {
    "grounding3d_openvocab_tiny": {"width": 24, "depth": 1},
    "grounding3d_openvocab_small": {"width": 32, "depth": 2},
    "grounding3d_openvocab_base": {"width": 48, "depth": 3},
}


def build_grounding3d_openvocab_open_vocabulary_3d_model(
    *, in_channels: int, variant: str = "grounding3d_openvocab_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_open_vocabulary_3d_model(
        family="grounding3d_openvocab",
        mode="grounding3d",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_open_vocabulary_3d_model(
        build_grounding3d_openvocab_open_vocabulary_3d_model, "grounding3d_openvocab_tiny"
    )
