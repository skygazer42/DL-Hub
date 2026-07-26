from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "vitan_caption_tiny": {"width": 24, "depth": 1},
    "vitan_caption_small": {"width": 32, "depth": 2},
    "vitan_caption_base": {"width": 48, "depth": 3},
}


def build_vitan_caption_video_captioner(
    *, in_channels: int, variant: str = "vitan_caption_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="vitan_caption",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_vitan_caption_video_captioner, "vitan_caption_tiny")
