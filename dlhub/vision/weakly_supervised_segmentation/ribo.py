from __future__ import annotations
from ._common import build_baseline_ws_segmenter, smoke_test_wss

_VARIANTS = {
    "ribo_tiny": {"width": 24, "depth": 1},
    "ribo_small": {"width": 32, "depth": 2},
    "ribo_base": {"width": 48, "depth": 3},
}


def build_ribo_ws_segmenter(
    *, in_channels: int, num_classes: int, variant: str = "ribo_small", width_mult: float = 1.0
):
    return build_baseline_ws_segmenter(
        family="ribo",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_wss(build_ribo_ws_segmenter, "ribo_tiny")
