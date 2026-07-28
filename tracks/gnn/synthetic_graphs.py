from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_graphs: int = 512
    num_nodes: int = 10
    batch_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _make_cycle_adj(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        j = (i + 1) % n
        a[i, j] = 1.0
        a[j, i] = 1.0
    return a


def _make_star_adj(n: int, center: int = 0) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if i == center:
            continue
        a[center, i] = 1.0
        a[i, center] = 1.0
    return a


def _make_node_features_from_adj(adj: np.ndarray) -> np.ndarray:
    # Two simple, interpretable features:
    # - normalized degree
    # - constant 1 (bias-like feature)
    deg = adj.sum(axis=1, keepdims=True).astype(np.float32)
    deg_norm = deg / max(1.0, float(adj.shape[0] - 1))
    ones = np.ones_like(deg_norm, dtype=np.float32)
    return np.concatenate([deg_norm, ones], axis=1)


class SyntheticGraphDataset:
    """Fixed-size synthetic graphs for graph-level binary classification.

    Labels:
      0 = cycle graph
      1 = star graph
    """

    def __init__(self, *, num_graphs: int, num_nodes: int, seed: int = 0) -> None:
        self.num_graphs = int(num_graphs)
        self.num_nodes = int(num_nodes)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        labels = np.zeros(self.num_graphs, dtype=np.int64)
        labels[self.num_graphs // 2 :] = 1
        rng.shuffle(labels)
        self._labels = labels

        self._cycle_adj = _make_cycle_adj(self.num_nodes)
        self._star_adj = _make_star_adj(self.num_nodes)

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int):
        label = int(self._labels[int(idx)])
        adj = self._star_adj if label == 1 else self._cycle_adj
        x = _make_node_features_from_adj(adj)
        return (x, adj), label


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    ds = SyntheticGraphDataset(num_graphs=config.num_graphs, num_nodes=config.num_nodes, seed=config.seed)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(config.val_fraction), seed=int(config.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        xs = []
        adjs = []
        ys = []
        for (x, adj), y in batch:
            xs.append(torch.from_numpy(x).to(torch.float32))
            adjs.append(torch.from_numpy(adj).to(torch.float32))
            ys.append(int(y))
        x_t = torch.stack(xs, dim=0)  # (B, N, F)
        adj_t = torch.stack(adjs, dim=0)  # (B, N, N)
        y_t = torch.tensor(ys, dtype=torch.long)
        return (x_t, adj_t), y_t

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader
