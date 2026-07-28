from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "recipe1m_clip_tiny": {"width": 24, "depth": 1},
    "recipe1m_clip_small": {"width": 32, "depth": 2},
    "recipe1m_clip_base": {"width": 48, "depth": 3},
}


def build_recipe1m_clip_food_classifier(
    *, in_channels: int, variant: str = "recipe1m_clip_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="recipe1m_clip",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_recipe1m_clip_food_classifier, "recipe1m_clip_tiny")
