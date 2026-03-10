"""Isomap (Isometric Mapping) manifold learning in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pairwise_euclidean_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sq_norms = np.sum(x**2, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2.0 * (x @ x.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.sqrt(sq_dists)


def _floyd_warshall(dist: np.ndarray) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be a square matrix")

    out = dist.copy()
    n = int(out.shape[0])
    for k in range(n):
        out = np.minimum(out, out[:, k, None] + out[k, None, :])
    return out


@dataclass
class Isomap:
    n_neighbors: int = 5
    n_components: int = 2

    def fit(self, x: np.ndarray) -> Isomap:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")

        n_samples = int(x.shape[0])
        if n_samples < 2:
            raise ValueError("x must contain at least 2 samples")

        n_neighbors = int(self.n_neighbors)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if n_neighbors >= n_samples:
            raise ValueError("n_neighbors must be < n_samples")

        n_components = int(self.n_components)
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if n_components > n_samples:
            raise ValueError("n_components must be <= n_samples")

        dists = _pairwise_euclidean_distances(x)
        np.fill_diagonal(dists, np.inf)

        neighbor_idx = np.argpartition(dists, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]

        graph = np.full((n_samples, n_samples), np.inf, dtype=np.float64)
        rows = np.arange(n_samples)[:, None]
        graph[rows, neighbor_idx] = dists[rows, neighbor_idx]
        np.fill_diagonal(graph, 0.0)

        graph = np.minimum(graph, graph.T)

        geodesic = _floyd_warshall(graph)
        if not np.all(np.isfinite(geodesic)):
            raise ValueError("Neighborhood graph is disconnected; increase n_neighbors.")

        self.geodesic_distances_ = geodesic

        d2 = geodesic**2
        row_mean = d2.mean(axis=1, keepdims=True)
        col_mean = d2.mean(axis=0, keepdims=True)
        total_mean = float(d2.mean())
        b = -0.5 * (d2 - row_mean - col_mean + total_mean)

        eigenvalues, eigenvectors = np.linalg.eigh(b)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        top = eigenvalues[:n_components]
        if np.any(top < -1e-10):
            raise ValueError("Not enough positive eigenvalues to embed at requested n_components.")
        top = np.maximum(top, 0.0)

        self.embedding_ = eigenvectors[:, :n_components] * np.sqrt(top[None, :])
        self.eigenvalues_ = top
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.embedding_
