
import torch
from torch import nn

from dlhub.vision.backbones._nas import Genotype, NASNetworkClassifier


_ENAS = Genotype(
    # ENAS discovered cells are typically similar to NASNet-like operations.
    normal=(
        ("sep_conv_3x3", 0),
        ("sep_conv_5x5", 1),
        ("avg_pool_3x3", 0),
        ("skip_connect", 1),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("max_pool_3x3", 0),
        ("sep_conv_3x3", 0),
    ),
    normal_concat=(2, 3, 4, 5),
    reduce=(
        ("max_pool_3x3", 0),
        ("sep_conv_5x5", 1),
        ("avg_pool_3x3", 0),
        ("sep_conv_3x3", 2),
        ("max_pool_3x3", 1),
        ("skip_connect", 2),
        ("sep_conv_3x3", 0),
        ("avg_pool_3x3", 0),
    ),
    reduce_concat=(2, 3, 4, 5),
)


class ENASClassifier(NASNetworkClassifier):
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
            genotype=_ENAS,
            init_channels=int(init_channels),
            num_cells=int(num_cells),
            stem_multiplier=int(stem_multiplier),
            width_mult=float(width_mult),
            drop_path=float(drop_path),
            dropout=float(dropout),
        )


_VARIANTS: dict[str, dict] = {
    "enas_tiny": {"init_channels": 12, "num_cells": 6, "stem_multiplier": 3},
    "enas_small": {"init_channels": 16, "num_cells": 8, "stem_multiplier": 3},
    "enas_base": {"init_channels": 24, "num_cells": 12, "stem_multiplier": 3},
}


def build_enas_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "enas_small",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ENAS variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ENASClassifier(
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
    m = build_enas_classifier(in_channels=3, num_classes=10, variant="enas_tiny", width_mult=0.5)
    y = m(x)
    print("enas_tiny", tuple(y.shape))

