from __future__ import annotations

from torch import nn

from ._common import build_baseline_text_to_video, smoke_test_text_to_video

_VARIANTS: dict[str, dict[str, int]] = {
    "lavie_t2v_tiny": {"width": 24, "depth": 1, "frames": 4},
    "lavie_t2v_small": {"width": 32, "depth": 2, "frames": 5},
    "lavie_t2v_base": {"width": 48, "depth": 3, "frames": 6},
}


def build_lavie_t2v_text_to_video(
    *, in_channels: int = 3, variant: str = "lavie_t2v_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_text_to_video(
        family="lavie_t2v",
        mode="lavie",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text_to_video(build_lavie_t2v_text_to_video, "lavie_t2v_tiny")
