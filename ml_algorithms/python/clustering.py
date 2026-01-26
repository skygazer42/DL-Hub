"""Additional clustering algorithms in NumPy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DBSCAN:
    eps: float = 0.5
    min_samples: int = 5

    def fit(self, x: np.ndarray) -> "DBSCAN":
        x = np.asarray(x, dtype=np.float64)
        n_samples = x.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        def region_query(idx: int) -> np.ndarray:
            distances = np.sqrt(((x - x[idx]) ** 2).sum(axis=1))
            return np.where(distances <= self.eps)[0]

        for idx in range(n_samples):
            if visited[idx]:
                continue
            visited[idx] = True
            neighbors = region_query(idx)
            if neighbors.size < self.min_samples:
                labels[idx] = -1
                continue
            labels[idx] = cluster_id
            seeds = list(neighbors)
            while seeds:
                current = seeds.pop()
                if not visited[current]:
                    visited[current] = True
                    current_neighbors = region_query(current)
                    if current_neighbors.size >= self.min_samples:
                        seeds.extend([n for n in current_neighbors if n not in seeds])
                if labels[current] == -1:
                    labels[current] = cluster_id
            cluster_id += 1
        self.labels_ = labels
        return self
