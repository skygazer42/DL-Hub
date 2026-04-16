from __future__ import annotations

from torch import nn

from ._common import build_toy_video_to_video, smoke_test_video_to_video

_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_v2v_tiny": {"width": 24, "depth": 1},
    "transformer_v2v_small": {"width": 32, "depth": 2},
    "transformer_v2v_base": {"width": 48, "depth": 3},
}


def build_transformer_v2v_video_to_video(
    *, in_channels: int = 3, variant: str = "transformer_v2v_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_video_to_video(
        family="transformer_v2v",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_video_to_video(build_transformer_v2v_video_to_video, "transformer_v2v_tiny")
