from __future__ import annotations
from ._common import build_toy_mesh, smoke_test_mesh

_VARIANTS = {
    "mambahmr_tiny": {"width": 24, "depth": 1},
    "mambahmr_small": {"width": 32, "depth": 2},
    "mambahmr_base": {"width": 48, "depth": 3},
}


def build_mambahmr_mesh_recoverer(
    *,
    in_channels: int,
    variant: str = "mambahmr_small",
    width_mult: float = 1.0,
    num_vertices: int = 32,
):
    return build_toy_mesh(
        family="mambahmr",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_vertices=int(num_vertices),
    )


if __name__ == "__main__":
    smoke_test_mesh(build_mambahmr_mesh_recoverer, "mambahmr_tiny")
