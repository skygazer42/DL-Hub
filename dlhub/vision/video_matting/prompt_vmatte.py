from __future__ import annotations

from ._common import build_baseline_video_matter, smoke_test_video_matter

_VARIANTS = {
    "prompt_vmatte_tiny": {"width": 24, "depth": 1},
    "prompt_vmatte_small": {"width": 32, "depth": 2},
    "prompt_vmatte_base": {"width": 48, "depth": 3},
}


def build_prompt_vmatte_video_matter(
    *, in_channels: int, variant: str = "prompt_vmatte_small", width_mult: float = 1.0
):
    return build_baseline_video_matter(
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_video_matter(build_prompt_vmatte_video_matter, "prompt_vmatte_tiny")
