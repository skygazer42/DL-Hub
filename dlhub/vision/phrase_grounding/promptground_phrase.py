from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "promptground_phrase_tiny": {"width": 24, "depth": 1},
    "promptground_phrase_small": {"width": 32, "depth": 2},
    "promptground_phrase_base": {"width": 48, "depth": 3},
}


def build_promptground_phrase_phrase_grounder(
    *,
    in_channels: int,
    variant: str = "promptground_phrase_small",
    width_mult: float = 1.0,
    **kwargs,
):
    return build_baseline_model(
        family="promptground_phrase",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_promptground_phrase_phrase_grounder, "promptground_phrase_tiny")
