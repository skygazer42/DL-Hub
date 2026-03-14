from __future__ import annotations

from ._common import MOTTracker2D, smoke_test_tracker

_VARIANTS: dict[str, dict[str, int]] = {
    "correlation_clustering_tiny": {"width": 64, "num_tracks": 32},
    "correlation_clustering_small": {"width": 96, "num_tracks": 48},
    "correlation_clustering_base": {"width": 128, "num_tracks": 64},
}


def build_correlation_clustering_tracker(
    *,
    in_channels: int,
    num_classes: int,
    seq_len: int = 4,
    image_size: int = 64,
    variant: str = "correlation_clustering_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
):
    cfg = _VARIANTS.get(str(variant).lower().strip())
    if cfg is None:
        raise ValueError(
            f"Unknown variant for correlation_clustering: {variant!r}. Available: {sorted(_VARIANTS)}"
        )

    # Keep seq_len/image_size in signature for a unified zoo build API.
    _ = seq_len, image_size

    width = max(16, int(round(int(cfg["width"]) * float(width_mult))))
    return MOTTracker2D(
        family="correlation_clustering",
        group="global_optimization",
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_tracks=int(cfg["num_tracks"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_tracker(build_correlation_clustering_tracker, "correlation_clustering_tiny")
