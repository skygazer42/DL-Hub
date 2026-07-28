from __future__ import annotations

from torch import nn

from ._common import build_baseline_video_to_video, smoke_test_video_to_video

_VARIANTS: dict[str, dict[str, int]] = {
    "style_vid2vid_tiny": {"width": 24, "depth": 1},
    "style_vid2vid_small": {"width": 32, "depth": 2},
    "style_vid2vid_base": {"width": 48, "depth": 3},
}


def build_style_vid2vid_video_to_video(
    *, in_channels: int = 3, variant: str = "style_vid2vid_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_video_to_video(
        family="style_vid2vid",
        mode="style",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_video_to_video(build_style_vid2vid_video_to_video, "style_vid2vid_tiny")
