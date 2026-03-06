from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dlhub.paths import get_repo_root

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class KarateGraph:
    num_nodes: int
    edge_index: torch.Tensor  # (2, E) int64, undirected (both directions)
    adj: torch.Tensor  # (N, N) float32, includes self-loops by default


def _karate_dir() -> Path:
    return get_repo_root() / "tracks" / "gnn" / "assets" / "karate"


def load_karate(*, dataset_dir: str | Path | None = None, add_self_loops: bool = True) -> KarateGraph:
    """Load the Karate Club edge list shipped with this repo."""

    import torch

    root = Path(dataset_dir) if dataset_dir is not None else _karate_dir()
    path = root / "karate.edgelist"

    edges: set[tuple[int, int]] = set()
    max_node = -1
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        a_str, b_str = line.split()
        a = int(a_str) - 1
        b = int(b_str) - 1
        max_node = max(max_node, a, b)

        # Undirected: store both directions and dedupe.
        edges.add((a, b))
        edges.add((b, a))

    num_nodes = max_node + 1
    if num_nodes <= 0:
        raise RuntimeError(f"Failed to infer num_nodes from {path}")

    edge_list = sorted(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # (2, E)

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    if add_self_loops:
        adj.fill_diagonal_(1.0)

    return KarateGraph(num_nodes=num_nodes, edge_index=edge_index, adj=adj)


__all__ = ["KarateGraph", "load_karate"]
