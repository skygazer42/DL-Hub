from __future__ import annotations
from torch import nn
from ._common import build_baseline_open_vocabulary_3d_model, smoke_test_open_vocabulary_3d_model

_VARIANTS: dict[str, dict[str, int]] = {
    "languagefield_3d_tiny": {"width": 24, "depth": 1},
    "languagefield_3d_small": {"width": 32, "depth": 2},
    "languagefield_3d_base": {"width": 48, "depth": 3},
}


def build_languagefield_3d_open_vocabulary_3d_model(
    *, in_channels: int, variant: str = "languagefield_3d_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_open_vocabulary_3d_model(
        family="languagefield_3d",
        mode="languagefield",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_open_vocabulary_3d_model(
        build_languagefield_3d_open_vocabulary_3d_model, "languagefield_3d_tiny"
    )
