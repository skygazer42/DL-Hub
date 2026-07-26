from __future__ import annotations
from torch import nn
from ._common import build_toy_reidentifier, smoke_test_reid

_VARIANTS = {
    "transreid_tiny": {"width": 24, "depth": 1, "embed": 96},
    "transreid_small": {"width": 32, "depth": 2, "embed": 128},
    "transreid_base": {"width": 48, "depth": 3, "embed": 160},
}


def build_transreid_reidentifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "transreid_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_toy_reidentifier(
        family="transreid",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        pooling="avg",
        part_branches=0,
    )


if __name__ == "__main__":
    smoke_test_reid(build_transreid_reidentifier, "transreid_tiny")
