from __future__ import annotations

from ._common import MOTTracker2D, smoke_test_tracker

_VARIANTS: dict[str, dict[str, int]] = {
    "megamot_tiny": {"width": 80, "num_tracks": 40},
    "megamot_small": {"width": 112, "num_tracks": 56},
    "megamot_base": {"width": 144, "num_tracks": 72},
}


def build_megamot_tracker(
    *,
    in_channels: int,
    num_classes: int,
    seq_len: int = 4,
    image_size: int = 64,
    variant: str = "megamot_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
):
    cfg = _VARIANTS.get(str(variant).lower().strip())
    if cfg is None:
        raise ValueError(
            f"Unknown variant for megamot: {variant!r}. Available: {sorted(_VARIANTS)}"
        )

    _ = seq_len, image_size
    width = max(16, int(round(int(cfg["width"]) * float(width_mult))))
    return MOTTracker2D(
        family="megamot",
        group="global_optimization",
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_tracks=int(cfg["num_tracks"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_tracker(build_megamot_tracker, "megamot_tiny")
