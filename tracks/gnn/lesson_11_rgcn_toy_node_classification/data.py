
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DataConfig:
    num_nodes: int = 180
    num_rels: int = 4
    num_classes: int = 3
    feature_dim: int = 16

    edges_per_node: int = 4
    feature_noise: float = 0.25

    train_fraction: float = 0.6
    val_fraction: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class ToyRelGraph:
    features: torch.Tensor  # (N, F)
    labels: torch.Tensor  # (N,)
    edge_index: torch.Tensor  # (2, E) directed
    edge_type: torch.Tensor  # (E,) in [0, R)
    edge_norm: torch.Tensor  # (E,) normalization for aggregation

    idx_train: torch.Tensor  # (N_train,)
    idx_val: torch.Tensor  # (N_val,)
    idx_test: torch.Tensor  # (N_test,)

    num_nodes: int
    num_rels: int
    num_classes: int


def _make_splits(num_nodes: int, train_fraction: float, val_fraction: float, *, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (0.0 < float(train_fraction) < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    if float(train_fraction) + float(val_fraction) >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    idx = torch.randperm(int(num_nodes), generator=gen)
    n_train = int(round(int(num_nodes) * float(train_fraction)))
    n_val = int(round(int(num_nodes) * float(val_fraction)))
    idx_train = idx[:n_train].to(torch.long)
    idx_val = idx[n_train : n_train + n_val].to(torch.long)
    idx_test = idx[n_train + n_val :].to(torch.long)
    return idx_train, idx_val, idx_test


def load_toy_rel_graph(cfg: DataConfig) -> ToyRelGraph:
    """Create a toy multi-relational directed graph.

    Generation idea:
    - labels define clusters
    - relation types connect nodes in different patterns w.r.t. labels
    """

    gen = torch.Generator().manual_seed(int(cfg.seed))
    n = int(cfg.num_nodes)
    r = int(cfg.num_rels)
    c = int(cfg.num_classes)
    f = int(cfg.feature_dim)

    labels = torch.randint(low=0, high=c, size=(n,), generator=gen, dtype=torch.long)

    # Features are noisy class prototypes.
    prototypes = torch.randn((c, f), generator=gen)
    features = prototypes[labels] + float(cfg.feature_noise) * torch.randn((n, f), generator=gen)

    # Build edges. For each node i and relation rel, sample neighbors from a target label bucket.
    # rel 0: same class
    # rel 1: next class (cyclic)
    # rel 2: prev class (cyclic)
    # rel 3+: random
    nodes_by_label: list[torch.Tensor] = []
    for y in range(c):
        nodes_by_label.append(torch.nonzero(labels == y, as_tuple=False).view(-1))

    src_list: list[int] = []
    dst_list: list[int] = []
    type_list: list[int] = []

    for i in range(n):
        yi = int(labels[i].item())
        for rel in range(r):
            if rel == 0:
                target_label = yi
            elif rel == 1:
                target_label = (yi + 1) % c
            elif rel == 2:
                target_label = (yi - 1) % c
            else:
                target_label = int(torch.randint(low=0, high=c, size=(1,), generator=gen).item())

            candidates = nodes_by_label[target_label]
            if int(candidates.numel()) == 0:
                continue
            # Sample a few edges per relation.
            for _ in range(int(cfg.edges_per_node)):
                j = int(candidates[torch.randint(0, int(candidates.numel()), (1,), generator=gen)].item())
                if j == i:
                    continue
                src_list.append(i)
                dst_list.append(j)
                type_list.append(rel)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(type_list, dtype=torch.long)
    e = int(edge_type.numel())
    if e == 0:
        raise RuntimeError("No edges generated; check config")

    # Simple normalization: 1 / out_degree(dst) (aggregating into dst).
    dst = edge_index[1]
    deg = torch.bincount(dst, minlength=n).to(torch.float32).clamp(min=1.0)
    edge_norm = (1.0 / deg[dst]).to(torch.float32)

    idx_train, idx_val, idx_test = _make_splits(
        n,
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
        gen=gen,
    )

    return ToyRelGraph(
        features=features,
        labels=labels,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_norm=edge_norm,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        num_nodes=n,
        num_rels=r,
        num_classes=c,
    )


__all__ = ["DataConfig", "ToyRelGraph", "load_toy_rel_graph"]

