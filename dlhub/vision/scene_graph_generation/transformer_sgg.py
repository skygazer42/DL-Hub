from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "transformer_sgg_tiny": {"width": 24, "depth": 1},
    "transformer_sgg_small": {"width": 32, "depth": 2},
    "transformer_sgg_base": {"width": 48, "depth": 3},
}


def build_transformer_sgg_scene_graph_model(
    *,
    in_channels: int,
    num_objects: int,
    num_relations: int,
    variant: str = "transformer_sgg_small",
    width_mult: float = 1.0,
):
    return build_baseline_model(
        family="transformer_sgg",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_objects=int(num_objects),
        num_relations=int(num_relations),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_transformer_sgg_scene_graph_model, "transformer_sgg_tiny")
