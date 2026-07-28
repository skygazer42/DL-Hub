from __future__ import annotations

from torch import nn

from ._common import build_baseline_road_scene_model, smoke_test_road_scene_model


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_road_scene_tiny": {"width": 24, "depth": 1},
    "mamba_road_scene_small": {"width": 36, "depth": 2},
    "mamba_road_scene_base": {"width": 48, "depth": 3},
}


def build_mamba_road_scene_road_scene_model(
    *,
    in_channels: int,
    variant: str = "mamba_road_scene_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_road_scene_model(
        family="mamba_road_scene",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_road_scene_model(build_mamba_road_scene_road_scene_model, "mamba_road_scene_tiny")
