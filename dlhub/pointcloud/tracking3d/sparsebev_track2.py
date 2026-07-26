from __future__ import annotations

from torch import nn

from ._common import KalmanAssociationTracker3D, smoke_test_tracker

_VARIANTS: dict[str, dict[str, int]] = {
    "sparsebev_track2_tiny": {"width": 48, "tracks": 24},
    "sparsebev_track2_small": {"width": 64, "tracks": 32},
    "sparsebev_track2_base": {"width": 96, "tracks": 48},
}


def build_sparsebev_track2_tracker3d(
    *,
    in_channels: int,
    num_classes: int,
    seq_len: int = 4,
    variant: str = "sparsebev_track2_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return KalmanAssociationTracker3D(
        family="sparsebev_track2",
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_tracks=int(cfg["tracks"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_tracker(build_sparsebev_track2_tracker3d, "sparsebev_track2_tiny")
