from __future__ import annotations

from torch import nn

from ._common import build_toy_road_scene_model, smoke_test_road_scene_model


_VARIANTS: dict[str, dict[str, int]] = {'prompt_road_scene_tiny': {'width': 24, 'depth': 1}, 'prompt_road_scene_small': {'width': 36, 'depth': 2}, 'prompt_road_scene_base': {'width': 48, 'depth': 3}}


def build_prompt_road_scene_road_scene_model(
    *,
    in_channels: int,
    variant: str = 'prompt_road_scene_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_road_scene_model(
        family='prompt_road_scene',
        mode='prompt',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_road_scene_model(build_prompt_road_scene_road_scene_model, 'prompt_road_scene_tiny')
