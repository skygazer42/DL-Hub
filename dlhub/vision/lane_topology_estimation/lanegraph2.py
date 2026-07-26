from __future__ import annotations
from ._common import build_toy_topology, smoke_test_topology

_VARIANTS = {
    "lanegraph2_tiny": {"width": 24, "depth": 1},
    "lanegraph2_small": {"width": 32, "depth": 2},
    "lanegraph2_base": {"width": 48, "depth": 3},
}


def build_lanegraph2_lane_topology_estimator(
    *,
    in_channels: int,
    variant: str = "lanegraph2_small",
    width_mult: float = 1.0,
    num_nodes: int = 8,
):
    return build_toy_topology(
        family="lanegraph2",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_nodes=int(num_nodes),
    )


if __name__ == "__main__":
    smoke_test_topology(build_lanegraph2_lane_topology_estimator, "lanegraph2_tiny")
