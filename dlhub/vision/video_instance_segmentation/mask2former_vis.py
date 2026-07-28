from __future__ import annotations
from ._common import build_baseline_vis, smoke_test_vis

_VARIANTS = {
    "mask2former_vis_tiny": {"width": 24, "depth": 1},
    "mask2former_vis_small": {"width": 32, "depth": 2},
    "mask2former_vis_base": {"width": 48, "depth": 3},
}


def build_mask2former_vis_video_instance_segmenter(
    *,
    in_channels: int,
    variant: str = "mask2former_vis_small",
    width_mult: float = 1.0,
    num_instances: int = 8,
):
    return build_baseline_vis(
        family="mask2former_vis",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_instances=int(num_instances),
    )


if __name__ == "__main__":
    smoke_test_vis(build_mask2former_vis_video_instance_segmenter, "mask2former_vis_tiny")
