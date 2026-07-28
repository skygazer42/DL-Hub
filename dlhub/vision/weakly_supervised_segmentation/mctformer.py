from __future__ import annotations
from ._common import build_baseline_ws_segmenter, smoke_test_wss

_VARIANTS = {
    "mctformer_tiny": {"width": 24, "depth": 1},
    "mctformer_small": {"width": 32, "depth": 2},
    "mctformer_base": {"width": 48, "depth": 3},
}


def build_mctformer_ws_segmenter(
    *, in_channels: int, num_classes: int, variant: str = "mctformer_small", width_mult: float = 1.0
):
    return build_baseline_ws_segmenter(
        family="mctformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_wss(build_mctformer_ws_segmenter, "mctformer_tiny")
