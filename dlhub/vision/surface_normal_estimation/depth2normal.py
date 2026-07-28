from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "depth2normal_tiny": {"width": 24, "depth": 1},
    "depth2normal_small": {"width": 32, "depth": 2},
    "depth2normal_base": {"width": 48, "depth": 3},
}


def build_depth2normal_normal_estimator(
    *, in_channels: int, variant: str = "depth2normal_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="depth2normal",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_depth2normal_normal_estimator, "depth2normal_tiny")
