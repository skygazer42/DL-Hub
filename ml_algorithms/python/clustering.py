"""Additional clustering algorithms in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DBSCAN:
    eps: float = 0.5
    min_samples: int = 5

    def fit(self, x: np.ndarray) -> DBSCAN:
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


@dataclass
class AgglomerativeClustering:
    n_clusters: int = 2
    linkage: str = "ward"

    def fit(self, x: np.ndarray) -> AgglomerativeClustering:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_samples = int(x.shape[0])
        if n_samples < 1:
            raise ValueError("x must be non-empty")
        if int(self.n_clusters) < 2:
            raise ValueError("n_clusters must be >= 2")
        if int(self.n_clusters) > n_samples:
            raise ValueError("n_clusters must be <= n_samples")

        linkage = str(self.linkage).lower()
        allowed = {"single", "complete", "average", "ward"}
        if linkage not in allowed:
            raise ValueError(
                f"Unsupported linkage {self.linkage!r}. Choose from {sorted(allowed)}."
            )

        clusters = [np.array([idx], dtype=int) for idx in range(n_samples)]
        sizes = np.ones((n_samples,), dtype=np.float64)

        if linkage == "ward":
            diff = x[:, None, :] - x[None, :, :]
            dist = 0.5 * np.sum(diff**2, axis=2)
        else:
            dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)

        while len(clusters) > int(self.n_clusters):
            i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
            if j < i:
                i, j = j, i

            mask = np.ones((dist.shape[0],), dtype=bool)
            mask[[i, j]] = False
            remaining = np.nonzero(mask)[0]

            d_i = dist[i, remaining]
            d_j = dist[j, remaining]

            if linkage == "single":
                new_dist = np.minimum(d_i, d_j)
            elif linkage == "complete":
                new_dist = np.maximum(d_i, d_j)
            elif linkage == "average":
                total = sizes[i] + sizes[j]
                new_dist = (sizes[i] * d_i + sizes[j] * d_j) / total
            else:
                n_i = sizes[i]
                n_j = sizes[j]
                n_k = sizes[remaining]
                d_ij = float(dist[i, j])
                new_dist = ((n_i + n_k) * d_i + (n_j + n_k) * d_j - n_k * d_ij) / (n_i + n_j + n_k)

            merged = np.sort(np.concatenate([clusters[i], clusters[j]]))
            new_size = float(sizes[i] + sizes[j])

            for idx in sorted((i, j), reverse=True):
                clusters.pop(idx)
            clusters.append(merged)

            sizes = np.delete(sizes, [j, i])
            sizes = np.append(sizes, new_size)

            dist = np.delete(dist, [j, i], axis=0)
            dist = np.delete(dist, [j, i], axis=1)

            new_col = new_dist[:, None]
            dist = np.column_stack([dist, new_col])
            new_row = np.append(new_dist, np.inf)[None, :]
            dist = np.vstack([dist, new_row])

        labels = np.empty((n_samples,), dtype=int)
        for label, members in enumerate(clusters):
            labels[members] = label
        self.labels_ = labels
        return self
