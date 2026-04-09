
from __future__ import annotations

from ._common import build_toy_vision_direction, smoke_test_direction

_VARIANTS = {
    'mamba_vid_det_tiny': {'width': 24, 'depth': 1},
    'mamba_vid_det_small': {'width': 32, 'depth': 2},
    'mamba_vid_det_base': {'width': 48, 'depth': 3},
}


def build_mamba_vid_det_video_detector(*, in_channels: int, variant: str = 'mamba_vid_det_small', width_mult: float = 1.0):
    return build_toy_vision_direction(
        family='mamba_vid_det',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == '__main__':
    smoke_test_direction(build_mamba_vid_det_video_detector, 'mamba_vid_det_tiny')
