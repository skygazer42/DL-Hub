from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._nas import Genotype, NASNetworkClassifier


_NASNET_A = Genotype(
    # A NASNet-A-like normal cell (simplified to DARTS-style genotype format).
    normal=(
        ("sep_conv_5x5", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_5x5", 1),
        ("sep_conv_3x3", 0),
        ("avg_pool_3x3", 1),
        ("skip_connect", 0),
        ("avg_pool_3x3", 0),
        ("avg_pool_3x3", 0),
    ),
    normal_concat=(2, 3, 4, 5),
    # Reduction cell biases toward pooling + factorized reductions.
    reduce=(
        ("max_pool_3x3", 0),
        ("sep_conv_5x5", 1),
        ("max_pool_3x3", 0),
        ("sep_conv_3x3", 2),
        ("avg_pool_3x3", 0),
        ("skip_connect", 2),
        ("sep_conv_3x3", 1),
        ("avg_pool_3x3", 0),
    ),
    reduce_concat=(2, 3, 4, 5),
)


class NASNetClassifier(NASNetworkClassifier):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        init_channels: int = 16,
        num_cells: int = 8,
        stem_multiplier: int = 3,
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            genotype=_NASNET_A,
            init_channels=int(init_channels),
            num_cells=int(num_cells),
            stem_multiplier=int(stem_multiplier),
            width_mult=float(width_mult),
            drop_path=float(drop_path),
            dropout=float(dropout),
        )


_VARIANTS: dict[str, dict] = {
    # Naming follows common NASNet family usage; "mobile" is smaller.
    "nasnet_mobile": {"init_channels": 12, "num_cells": 6, "stem_multiplier": 3},
    "nasnet_small": {"init_channels": 16, "num_cells": 8, "stem_multiplier": 3},
    "nasnet_large": {"init_channels": 24, "num_cells": 12, "stem_multiplier": 3},
}


def build_nasnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "nasnet_mobile",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown NASNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return NASNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        init_channels=int(spec["init_channels"]),
        num_cells=int(spec["num_cells"]),
        stem_multiplier=int(spec["stem_multiplier"]),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_nasnet_classifier(in_channels=3, num_classes=10, variant="nasnet_mobile", width_mult=0.5)
    y = m(x)
    print("nasnet_mobile", tuple(y.shape))

