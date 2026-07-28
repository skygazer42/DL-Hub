from __future__ import annotations
from ._common import build_baseline_ws_detector, smoke_test_ws

_VARIANTS = {
    "tokendet_ws_tiny": {"width": 24, "depth": 1},
    "tokendet_ws_small": {"width": 32, "depth": 2},
    "tokendet_ws_base": {"width": 48, "depth": 3},
}


def build_tokendet_ws_ws_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "tokendet_ws_small",
    width_mult: float = 1.0,
):
    return build_baseline_ws_detector(
        family="tokendet_ws",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_ws(build_tokendet_ws_ws_detector, "tokendet_ws_tiny")
