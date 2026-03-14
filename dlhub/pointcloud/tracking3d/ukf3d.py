from __future__ import annotations

from torch import nn

from ._common import KalmanAssociationTracker3D, smoke_test_tracker

_VARIANTS: dict[str, dict[str, int]] = {
    "ukf3d_tiny": {"width": 48, "tracks": 24},
    "ukf3d_small": {"width": 64, "tracks": 32},
    "ukf3d_base": {"width": 96, "tracks": 48},
}


def build_ukf3d_tracker3d(
    *,
    in_channels: int,
    num_classes: int,
    seq_len: int = 4,
    variant: str = "ukf3d_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return KalmanAssociationTracker3D(
        family="ukf3d",
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_tracks=int(cfg["tracks"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_tracker(build_ukf3d_tracker3d, "ukf3d_tiny")
