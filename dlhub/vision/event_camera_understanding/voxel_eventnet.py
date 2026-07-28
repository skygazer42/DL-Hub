from __future__ import annotations

from ._common import build_baseline_event_model, smoke_test_event_model

_VARIANTS = {
    "voxel_eventnet_tiny": {"width": 24, "depth": 1, "steps": 3},
    "voxel_eventnet_small": {"width": 32, "depth": 2, "steps": 4},
    "voxel_eventnet_base": {"width": 48, "depth": 3, "steps": 5},
}


def build_voxel_eventnet_event_model(
    *, in_channels: int, variant: str = "voxel_eventnet_small", width_mult: float = 1.0
):
    return build_baseline_event_model(
        family="voxel_eventnet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_event_model(build_voxel_eventnet_event_model, "voxel_eventnet_tiny")
