from __future__ import annotations
from ._common import build_baseline_topology, smoke_test_topology

_VARIANTS = {
    "bevlanegraph_tiny": {"width": 24, "depth": 1},
    "bevlanegraph_small": {"width": 32, "depth": 2},
    "bevlanegraph_base": {"width": 48, "depth": 3},
}


def build_bevlanegraph_lane_topology_estimator(
    *,
    in_channels: int,
    variant: str = "bevlanegraph_small",
    width_mult: float = 1.0,
    num_nodes: int = 8,
):
    return build_baseline_topology(
        family="bevlanegraph",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_nodes=int(num_nodes),
    )


if __name__ == "__main__":
    smoke_test_topology(build_bevlanegraph_lane_topology_estimator, "bevlanegraph_tiny")
