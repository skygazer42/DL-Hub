from __future__ import annotations
from ._common import build_baseline_inter, smoke_test_inter

_VARIANTS = {
    "pointprompt_seg_tiny": {"width": 24, "depth": 1},
    "pointprompt_seg_small": {"width": 32, "depth": 2},
    "pointprompt_seg_base": {"width": 48, "depth": 3},
}


def build_pointprompt_seg_interactive_segmenter(
    *, in_channels: int, variant: str = "pointprompt_seg_small", width_mult: float = 1.0
):
    return build_baseline_inter(
        family="pointprompt_seg",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_inter(build_pointprompt_seg_interactive_segmenter, "pointprompt_seg_tiny")
