from __future__ import annotations

from torch import nn

from ._common import BEVTracking3D, smoke_test_tracker

_VARIANTS: dict[str, dict[str, int]] = {
    "pillartrack_tiny": {"width": 64, "tracks": 24},
    "pillartrack_small": {"width": 96, "tracks": 32},
    "pillartrack_base": {"width": 128, "tracks": 48},
}


def build_pillartrack_tracker3d(
    *,
    in_channels: int,
    num_classes: int,
    seq_len: int = 4,
    variant: str = "pillartrack_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return BEVTracking3D(
        family="pillartrack",
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_tracks=int(cfg["tracks"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_tracker(build_pillartrack_tracker3d, "pillartrack_tiny")
