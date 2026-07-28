from __future__ import annotations

from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "deep_relight_tiny": {"width": 24, "depth": 1},
    "deep_relight_small": {"width": 32, "depth": 2},
    "deep_relight_base": {"width": 48, "depth": 3},
}


def build_deep_relight_relighter(
    *, in_channels: int, variant: str = "deep_relight_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="deep_relight",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_deep_relight_relighter, "deep_relight_tiny")
