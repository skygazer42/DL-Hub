from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "mambasgg_tiny": {"width": 24, "depth": 1},
    "mambasgg_small": {"width": 32, "depth": 2},
    "mambasgg_base": {"width": 48, "depth": 3},
}


def build_mambasgg_scene_graph_model(
    *,
    in_channels: int,
    num_objects: int,
    num_relations: int,
    variant: str = "mambasgg_small",
    width_mult: float = 1.0,
):
    return build_baseline_model(
        family="mambasgg",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_objects=int(num_objects),
        num_relations=int(num_relations),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_mambasgg_scene_graph_model, "mambasgg_tiny")
