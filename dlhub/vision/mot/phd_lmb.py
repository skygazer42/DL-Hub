from __future__ import annotations

from ._common import MOTTracker2D, smoke_test_tracker

_VARIANTS: dict[str, dict[str, int]] = {
    "phd_lmb_tiny": {"width": 64, "num_tracks": 32},
    "phd_lmb_small": {"width": 96, "num_tracks": 48},
    "phd_lmb_base": {"width": 128, "num_tracks": 64},
}


def build_phd_lmb_tracker(
    *,
    in_channels: int,
    num_classes: int,
    seq_len: int = 4,
    image_size: int = 64,
    variant: str = "phd_lmb_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
):
    cfg = _VARIANTS.get(str(variant).lower().strip())
    if cfg is None:
        raise ValueError(
            f"Unknown variant for phd_lmb: {variant!r}. Available: {sorted(_VARIANTS)}"
        )

    _ = seq_len, image_size
    width = max(16, int(round(int(cfg["width"]) * float(width_mult))))
    return MOTTracker2D(
        family="phd_lmb",
        group="probabilistic_filtering",
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_tracks=int(cfg["num_tracks"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_tracker(build_phd_lmb_tracker, "phd_lmb_tiny")
